# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Load dataset
diamonds = sns.load_dataset("diamonds")

# Separate features and target
X = diamonds.drop("price", axis=1)
y = diamonds["price"]

# Convert categorical columns to 'category' dtype
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = X[col].astype("category")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# Set parameters (change tree_method to "gpu_hist" if using GPU)
params = {
    "objective": "reg:squarederror",
   "tree_method": "hist",  # Use "hist" if you don't have GPU

    #"tree_method": "gpu_hist" if xgb.get_config()["use_cuda"] else "hist",
    "eval_metric": "rmse", # metric to minimize
    'max_depth': 6, # limt the tree depth 
    'eta': 0.05,    # learning rate
    'min_child_weight': 3,  # minimum sum of instance weight needed in child 
    'subsample': 0.8,   # fraction of training data to use per tree
    'colsample_bytree': 0.8 #fraction of features to consider per tree

}

# Train the model with evaluation
evals = [(dtrain, "train"), (dtest, "test")]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=10
)

# Predict and evaluate
preds = model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse:.2f}")

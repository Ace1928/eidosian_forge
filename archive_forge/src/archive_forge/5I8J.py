from pdpbox import pdp
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels

# Train a random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Select the features to visualize
features = ["feature1", "feature2"]

# Create the partial dependence plots
pdp_iso = pdp.pdp_isolate(
    model=rf_regressor,
    dataset=X_train,
    model_features=X_train.columns,
    feature=features,
)

# Visualize the partial dependence plots
fig, axes = pdp.pdp_plot(pdp_iso, features, figsize=(10, 5))

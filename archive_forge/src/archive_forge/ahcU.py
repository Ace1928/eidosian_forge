from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load the dataset
X = ...  # Features
y = ...  # Target labels

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate permutation importance
result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

# Get the feature importances
importances = result.importances_mean

# Print the feature importances
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")

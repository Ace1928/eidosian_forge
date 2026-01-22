from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels
X_test = ...  # Testing features
y_test = ...  # Testing labels
# Train an SVM classifier
svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train, y_train)
# Calibrate the classifier using Platt scaling
calibrated_classifier = CalibratedClassifierCV(
    base_estimator=svm_classifier, method="sigmoid", cv=5
)
calibrated_classifier.fit(X_train, y_train)
# Make predictions on the testing data
y_pred_proba = calibrated_classifier.predict_proba(X_test)
# Evaluate the calibration using Brier score
brier_score = brier_score_loss(y_test, y_pred_proba[:, 1])
print(f"Brier Score: {brier_score:.3f}")

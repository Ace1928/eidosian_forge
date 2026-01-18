from collections import Counter
import numpy as np
import pytest
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('constructor_name, default_label', [('from_estimator', 'LogisticRegression (AP = {:.2f})'), ('from_predictions', 'Classifier (AP = {:.2f})')])
def test_precision_recall_display_name(pyplot, constructor_name, default_label):
    """Check the behaviour of the name parameters"""
    X, y = make_classification(n_classes=2, n_samples=100, random_state=0)
    pos_label = 1
    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)
    y_pred = classifier.predict_proba(X)[:, pos_label]
    assert constructor_name in ('from_estimator', 'from_predictions')
    if constructor_name == 'from_estimator':
        display = PrecisionRecallDisplay.from_estimator(classifier, X, y)
    else:
        display = PrecisionRecallDisplay.from_predictions(y, y_pred, pos_label=pos_label)
    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)
    assert display.line_.get_label() == default_label.format(average_precision)
    display.plot(name='MySpecialEstimator')
    assert display.line_.get_label() == f'MySpecialEstimator (AP = {average_precision:.2f})'
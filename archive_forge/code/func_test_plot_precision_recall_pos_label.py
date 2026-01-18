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
@pytest.mark.parametrize('constructor_name', ['from_estimator', 'from_predictions'])
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function'])
def test_plot_precision_recall_pos_label(pyplot, constructor_name, response_method):
    X, y = load_breast_cancer(return_X_y=True)
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = (X[idx_selected], y[idx_selected])
    X, y = shuffle(X, y, random_state=42)
    X = X[:, :2]
    y = np.array(['cancer' if c == 1 else 'not cancer' for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    assert classifier.classes_.tolist() == ['cancer', 'not cancer']
    y_pred = getattr(classifier, response_method)(X_test)
    y_pred_cancer = -1 * y_pred if y_pred.ndim == 1 else y_pred[:, 0]
    y_pred_not_cancer = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    if constructor_name == 'from_estimator':
        display = PrecisionRecallDisplay.from_estimator(classifier, X_test, y_test, pos_label='cancer', response_method=response_method)
    else:
        display = PrecisionRecallDisplay.from_predictions(y_test, y_pred_cancer, pos_label='cancer')
    avg_prec_limit = 0.65
    assert display.average_precision < avg_prec_limit
    assert -trapezoid(display.precision, display.recall) < avg_prec_limit
    if constructor_name == 'from_estimator':
        display = PrecisionRecallDisplay.from_estimator(classifier, X_test, y_test, response_method=response_method, pos_label='not cancer')
    else:
        display = PrecisionRecallDisplay.from_predictions(y_test, y_pred_not_cancer, pos_label='not cancer')
    avg_prec_limit = 0.95
    assert display.average_precision > avg_prec_limit
    assert -trapezoid(display.precision, display.recall) > avg_prec_limit
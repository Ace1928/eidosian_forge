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
@pytest.mark.parametrize('drop_intermediate', [True, False])
def test_precision_recall_display_plotting(pyplot, constructor_name, response_method, drop_intermediate):
    """Check the overall plotting rendering."""
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    pos_label = 1
    classifier = LogisticRegression().fit(X, y)
    classifier.fit(X, y)
    y_pred = getattr(classifier, response_method)(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, pos_label]
    assert constructor_name in ('from_estimator', 'from_predictions')
    if constructor_name == 'from_estimator':
        display = PrecisionRecallDisplay.from_estimator(classifier, X, y, response_method=response_method, drop_intermediate=drop_intermediate)
    else:
        display = PrecisionRecallDisplay.from_predictions(y, y_pred, pos_label=pos_label, drop_intermediate=drop_intermediate)
    precision, recall, _ = precision_recall_curve(y, y_pred, pos_label=pos_label, drop_intermediate=drop_intermediate)
    average_precision = average_precision_score(y, y_pred, pos_label=pos_label)
    np.testing.assert_allclose(display.precision, precision)
    np.testing.assert_allclose(display.recall, recall)
    assert display.average_precision == pytest.approx(average_precision)
    import matplotlib as mpl
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert isinstance(display.figure_, mpl.figure.Figure)
    assert display.ax_.get_xlabel() == 'Recall (Positive label: 1)'
    assert display.ax_.get_ylabel() == 'Precision (Positive label: 1)'
    assert display.ax_.get_adjustable() == 'box'
    assert display.ax_.get_aspect() in ('equal', 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    display.plot(alpha=0.8, name='MySpecialEstimator')
    expected_label = f'MySpecialEstimator (AP = {average_precision:0.2f})'
    assert display.line_.get_label() == expected_label
    assert display.line_.get_alpha() == pytest.approx(0.8)
    assert display.chance_level_ is None
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import trapezoid
@pytest.mark.parametrize('response_method', ['predict_proba', 'decision_function'])
@pytest.mark.parametrize('with_sample_weight', [True, False])
@pytest.mark.parametrize('drop_intermediate', [True, False])
@pytest.mark.parametrize('with_strings', [True, False])
@pytest.mark.parametrize('constructor_name, default_name', [('from_estimator', 'LogisticRegression'), ('from_predictions', 'Classifier')])
def test_roc_curve_display_plotting(pyplot, response_method, data_binary, with_sample_weight, drop_intermediate, with_strings, constructor_name, default_name):
    """Check the overall plotting behaviour."""
    X, y = data_binary
    pos_label = None
    if with_strings:
        y = np.array(['c', 'b'])[y]
        pos_label = 'c'
    if with_sample_weight:
        rng = np.random.RandomState(42)
        sample_weight = rng.randint(1, 4, size=X.shape[0])
    else:
        sample_weight = None
    lr = LogisticRegression()
    lr.fit(X, y)
    y_pred = getattr(lr, response_method)(X)
    y_pred = y_pred if y_pred.ndim == 1 else y_pred[:, 1]
    if constructor_name == 'from_estimator':
        display = RocCurveDisplay.from_estimator(lr, X, y, sample_weight=sample_weight, drop_intermediate=drop_intermediate, pos_label=pos_label, alpha=0.8)
    else:
        display = RocCurveDisplay.from_predictions(y, y_pred, sample_weight=sample_weight, drop_intermediate=drop_intermediate, pos_label=pos_label, alpha=0.8)
    fpr, tpr, _ = roc_curve(y, y_pred, sample_weight=sample_weight, drop_intermediate=drop_intermediate, pos_label=pos_label)
    assert_allclose(display.roc_auc, auc(fpr, tpr))
    assert_allclose(display.fpr, fpr)
    assert_allclose(display.tpr, tpr)
    assert display.estimator_name == default_name
    import matplotlib as mpl
    assert isinstance(display.line_, mpl.lines.Line2D)
    assert display.line_.get_alpha() == 0.8
    assert isinstance(display.ax_, mpl.axes.Axes)
    assert isinstance(display.figure_, mpl.figure.Figure)
    assert display.ax_.get_adjustable() == 'box'
    assert display.ax_.get_aspect() in ('equal', 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    expected_label = f'{default_name} (AUC = {display.roc_auc:.2f})'
    assert display.line_.get_label() == expected_label
    expected_pos_label = 1 if pos_label is None else pos_label
    expected_ylabel = f'True Positive Rate (Positive label: {expected_pos_label})'
    expected_xlabel = f'False Positive Rate (Positive label: {expected_pos_label})'
    assert display.ax_.get_ylabel() == expected_ylabel
    assert display.ax_.get_xlabel() == expected_xlabel
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
@pytest.mark.parametrize('class_method', ['from_estimator', 'from_predictions'])
def test_plot_prediction_error_ax(pyplot, regressor_fitted, class_method):
    """Check that we can pass an axis to the display."""
    _, ax = pyplot.subplots()
    if class_method == 'from_estimator':
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y, ax=ax)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, ax=ax)
    assert display.ax_ is ax
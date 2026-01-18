import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
@pytest.mark.parametrize('class_method', ['from_estimator', 'from_predictions'])
@pytest.mark.parametrize('kind', ['actual_vs_predicted', 'residual_vs_predicted'])
def test_prediction_error_display(pyplot, regressor_fitted, class_method, kind):
    """Check the default behaviour of the display."""
    if class_method == 'from_estimator':
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y, kind=kind)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, kind=kind)
    if kind == 'actual_vs_predicted':
        assert_allclose(display.line_.get_xdata(), display.line_.get_ydata())
        assert display.ax_.get_xlabel() == 'Predicted values'
        assert display.ax_.get_ylabel() == 'Actual values'
        assert display.line_ is not None
    else:
        assert display.ax_.get_xlabel() == 'Predicted values'
        assert display.ax_.get_ylabel() == 'Residuals (actual - predicted)'
        assert display.line_ is not None
    assert display.ax_.get_legend() is None
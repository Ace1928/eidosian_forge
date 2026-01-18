import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
@pytest.mark.parametrize('class_method', ['from_estimator', 'from_predictions'])
@pytest.mark.parametrize('subsample, expected_size', [(5, 5), (0.1, int(X.shape[0] * 0.1)), (None, X.shape[0])])
def test_plot_prediction_error_subsample(pyplot, regressor_fitted, class_method, subsample, expected_size):
    """Check the behaviour of `subsample`."""
    if class_method == 'from_estimator':
        display = PredictionErrorDisplay.from_estimator(regressor_fitted, X, y, subsample=subsample)
    else:
        y_pred = regressor_fitted.predict(X)
        display = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, subsample=subsample)
    assert len(display.scatter_.get_offsets()) == expected_size
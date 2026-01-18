import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_diabetes
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.metrics import PredictionErrorDisplay
@pytest.mark.parametrize('regressor, params, err_type, err_msg', [(Ridge().fit(X, y), {'subsample': -1}, ValueError, 'When an integer, subsample=-1 should be'), (Ridge().fit(X, y), {'subsample': 20.0}, ValueError, 'When a floating-point, subsample=20.0 should be'), (Ridge().fit(X, y), {'subsample': -20.0}, ValueError, 'When a floating-point, subsample=-20.0 should be'), (Ridge().fit(X, y), {'kind': 'xxx'}, ValueError, '`kind` must be one of')])
@pytest.mark.parametrize('class_method', ['from_estimator', 'from_predictions'])
def test_prediction_error_display_raise_error(pyplot, class_method, regressor, params, err_type, err_msg):
    """Check that we raise the proper error when making the parameters
    # validation."""
    with pytest.raises(err_type, match=err_msg):
        if class_method == 'from_estimator':
            PredictionErrorDisplay.from_estimator(regressor, X, y, **params)
        else:
            y_pred = regressor.predict(X)
            PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred, **params)
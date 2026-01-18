from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
@pytest.mark.filterwarnings("ignore:'squared' is deprecated")
@pytest.mark.parametrize('old_func, new_func', [(mean_squared_error, root_mean_squared_error), (mean_squared_log_error, root_mean_squared_log_error)])
def test_rmse_rmsle_parameter(old_func, new_func):
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])
    y_true = np.array([[0.5, 1], [1, 2], [7, 6]])
    y_pred = np.array([[0.5, 2], [1, 2.5], [8, 8]])
    sw = np.arange(len(y_true))
    expected = old_func(y_true, y_pred, squared=False)
    actual = new_func(y_true, y_pred)
    assert_allclose(expected, actual)
    expected = old_func(y_true, y_pred, sample_weight=sw, squared=False)
    actual = new_func(y_true, y_pred, sample_weight=sw)
    assert_allclose(expected, actual)
    expected = old_func(y_true, y_pred, multioutput='raw_values', squared=False)
    actual = new_func(y_true, y_pred, multioutput='raw_values')
    assert_allclose(expected, actual)
    expected = old_func(y_true, y_pred, sample_weight=sw, multioutput='raw_values', squared=False)
    actual = new_func(y_true, y_pred, sample_weight=sw, multioutput='raw_values')
    assert_allclose(expected, actual)
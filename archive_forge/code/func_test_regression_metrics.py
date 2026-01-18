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
def test_regression_metrics(n_samples=50):
    y_true = np.arange(n_samples)
    y_pred = y_true + 1
    y_pred_2 = y_true - 1
    assert_almost_equal(mean_squared_error(y_true, y_pred), 1.0)
    assert_almost_equal(mean_squared_log_error(y_true, y_pred), mean_squared_error(np.log(1 + y_true), np.log(1 + y_pred)))
    assert_almost_equal(mean_absolute_error(y_true, y_pred), 1.0)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred), 0.5)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2), 0.5)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred, alpha=0.4), 0.6)
    assert_almost_equal(mean_pinball_loss(y_true, y_pred_2, alpha=0.4), 0.4)
    assert_almost_equal(median_absolute_error(y_true, y_pred), 1.0)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(mape)
    assert mape > 1000000.0
    assert_almost_equal(max_error(y_true, y_pred), 1.0)
    assert_almost_equal(r2_score(y_true, y_pred), 0.995, 2)
    assert_almost_equal(r2_score(y_true, y_pred, force_finite=False), 0.995, 2)
    assert_almost_equal(explained_variance_score(y_true, y_pred), 1.0)
    assert_almost_equal(explained_variance_score(y_true, y_pred, force_finite=False), 1.0)
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=0), mean_squared_error(y_true, y_pred))
    assert_almost_equal(d2_tweedie_score(y_true, y_pred, power=0), r2_score(y_true, y_pred))
    dev_median = np.abs(y_true - np.median(y_true)).sum()
    assert_array_almost_equal(d2_absolute_error_score(y_true, y_pred), 1 - np.abs(y_true - y_pred).sum() / dev_median)
    alpha = 0.2
    pinball_loss = lambda y_true, y_pred, alpha: alpha * np.maximum(y_true - y_pred, 0) + (1 - alpha) * np.maximum(y_pred - y_true, 0)
    y_quantile = np.percentile(y_true, q=alpha * 100)
    assert_almost_equal(d2_pinball_score(y_true, y_pred, alpha=alpha), 1 - pinball_loss(y_true, y_pred, alpha).sum() / pinball_loss(y_true, y_quantile, alpha).sum())
    assert_almost_equal(d2_absolute_error_score(y_true, y_pred), d2_pinball_score(y_true, y_pred, alpha=0.5))
    y_true = np.arange(1, 1 + n_samples)
    y_pred = 2 * y_true
    n = n_samples
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=-1), 5 / 12 * n * (n ** 2 + 2 * n + 1))
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=1), (n + 1) * (1 - np.log(2)))
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=2), 2 * np.log(2) - 1)
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=3 / 2), (6 * np.sqrt(2) - 8) / n * np.sqrt(y_true).sum())
    assert_almost_equal(mean_tweedie_deviance(y_true, y_pred, power=3), np.sum(1 / y_true) / (4 * n))
    dev_mean = 2 * np.mean(xlogy(y_true, 2 * y_true / (n + 1)))
    assert_almost_equal(d2_tweedie_score(y_true, y_pred, power=1), 1 - (n + 1) * (1 - np.log(2)) / dev_mean)
    dev_mean = 2 * np.log((n + 1) / 2) - 2 / n * np.log(factorial(n))
    assert_almost_equal(d2_tweedie_score(y_true, y_pred, power=2), 1 - (2 * np.log(2) - 1) / dev_mean)
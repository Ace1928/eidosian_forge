import numpy as np
import pytest
from pytest import approx
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.utils._testing import assert_allclose, skip_if_32bit
from sklearn.utils.fixes import (
@pytest.mark.parametrize('q', [0.5, 0.9, 0.05])
def test_quantile_estimates_calibration(q, default_solver):
    X, y = make_regression(n_samples=1000, n_features=20, random_state=0, noise=1.0)
    quant = QuantileRegressor(quantile=q, alpha=0, solver=default_solver).fit(X, y)
    assert np.mean(y < quant.predict(X)) == approx(q, abs=0.01)
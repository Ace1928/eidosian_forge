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
def test_quantile_sample_weight(default_solver):
    n = 1000
    X, y = make_regression(n_samples=n, n_features=5, random_state=0, noise=10.0)
    weight = np.ones(n)
    weight[y > y.mean()] = 100
    quant = QuantileRegressor(quantile=0.5, alpha=1e-08, solver=default_solver)
    quant.fit(X, y, sample_weight=weight)
    fraction_below = np.mean(y < quant.predict(X))
    assert fraction_below > 0.5
    weighted_fraction_below = np.average(y < quant.predict(X), weights=weight)
    assert weighted_fraction_below == approx(0.5, abs=0.03)
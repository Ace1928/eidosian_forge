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
@pytest.mark.parametrize('quantile, alpha, intercept, coef', [[0.5, 0, 1, None], [0.51, 0, 1, 10], [0.49, 0, 1, 1], [0.5, 0.01, 1, 1], [0.5, 100, 2, 0]])
def test_quantile_toy_example(quantile, alpha, intercept, coef, default_solver):
    X = [[0], [1], [1]]
    y = [1, 2, 11]
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver=default_solver).fit(X, y)
    assert_allclose(model.intercept_, intercept, atol=0.01)
    if coef is not None:
        assert_allclose(model.coef_[0], coef, atol=0.01)
    if alpha < 100:
        assert model.coef_[0] >= 1
    assert model.coef_[0] <= 10
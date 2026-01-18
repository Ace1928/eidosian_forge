import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('return_intercept', [False, True])
@pytest.mark.parametrize('sample_weight', [None, np.ones(1000)])
@pytest.mark.parametrize('container', [np.array] + CSR_CONTAINERS)
@pytest.mark.parametrize('solver', ['auto', 'sparse_cg', 'cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'])
def test_ridge_regression_check_arguments_validity(return_intercept, sample_weight, container, solver):
    """check if all combinations of arguments give valid estimations"""
    rng = check_random_state(42)
    X = rng.rand(1000, 3)
    true_coefs = [1, 2, 0.1]
    y = np.dot(X, true_coefs)
    true_intercept = 0.0
    if return_intercept:
        true_intercept = 10000.0
    y += true_intercept
    X_testing = container(X)
    alpha, tol = (0.001, 1e-06)
    atol = 0.001 if _IS_32BIT else 0.0001
    positive = solver == 'lbfgs'
    if solver not in ['sag', 'auto'] and return_intercept:
        with pytest.raises(ValueError, match="In Ridge, only 'sag' solver"):
            ridge_regression(X_testing, y, alpha=alpha, solver=solver, sample_weight=sample_weight, return_intercept=return_intercept, positive=positive, tol=tol)
        return
    out = ridge_regression(X_testing, y, alpha=alpha, solver=solver, sample_weight=sample_weight, positive=positive, return_intercept=return_intercept, tol=tol)
    if return_intercept:
        coef, intercept = out
        assert_allclose(coef, true_coefs, rtol=0, atol=atol)
        assert_allclose(intercept, true_intercept, rtol=0, atol=atol)
    else:
        assert_allclose(out, true_coefs, rtol=0, atol=atol)
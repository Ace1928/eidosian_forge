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
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('alpha', [1.0, 0.01])
def test_ridge_regression_sample_weights(solver, fit_intercept, sparse_container, alpha, ols_ridge_dataset, global_random_seed):
    """Test that Ridge with sample weights gives correct results.

    We use the following trick:
        ||y - Xw||_2 = (z - Aw)' W (z - Aw)
    for z=[y, y], A' = [X', X'] (vstacked), and W[:n/2] + W[n/2:] = 1, W=diag(W)
    """
    if sparse_container is not None:
        if fit_intercept and solver not in SPARSE_SOLVERS_WITH_INTERCEPT:
            pytest.skip()
        elif not fit_intercept and solver not in SPARSE_SOLVERS_WITHOUT_INTERCEPT:
            pytest.skip()
    X, y, _, coef = ols_ridge_dataset
    n_samples, n_features = X.shape
    sw = rng.uniform(low=0, high=1, size=n_samples)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=1e-15 if solver in ['sag', 'saga'] else 1e-10, max_iter=100000, random_state=global_random_seed)
    X = X[:, :-1]
    X = np.concatenate((X, X), axis=0)
    y = np.r_[y, y]
    sw = np.r_[sw, 1 - sw] * alpha
    if fit_intercept:
        intercept = coef[-1]
    else:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        intercept = 0
    if sparse_container is not None:
        X = sparse_container(X)
    model.fit(X, y, sample_weight=sw)
    coef = coef[:-1]
    assert model.intercept_ == pytest.approx(intercept)
    assert_allclose(model.coef_, coef)
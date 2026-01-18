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
@pytest.mark.parametrize('solver', ['lsqr', 'sparse_cg', 'lbfgs', 'auto'])
@pytest.mark.parametrize('with_sample_weight', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse(solver, with_sample_weight, global_random_seed, csr_container):
    """Check that ridge finds the same coefs and intercept on dense and sparse input
    in the presence of sample weights.

    For now only sparse_cg and lbfgs can correctly fit an intercept
    with sparse X with default tol and max_iter.
    'sag' is tested separately in test_ridge_fit_intercept_sparse_sag because it
    requires more iterations and should raise a warning if default max_iter is used.
    Other solvers raise an exception, as checked in
    test_ridge_fit_intercept_sparse_error
    """
    positive = solver == 'lbfgs'
    X, y = _make_sparse_offset_regression(n_features=20, random_state=global_random_seed, positive=positive)
    sample_weight = None
    if with_sample_weight:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = 1.0 + rng.uniform(size=X.shape[0])
    dense_solver = 'sparse_cg' if solver == 'auto' else solver
    dense_ridge = Ridge(solver=dense_solver, tol=1e-12, positive=positive)
    sparse_ridge = Ridge(solver=solver, tol=1e-12, positive=positive)
    dense_ridge.fit(X, y, sample_weight=sample_weight)
    sparse_ridge.fit(csr_container(X), y, sample_weight=sample_weight)
    assert_allclose(dense_ridge.intercept_, sparse_ridge.intercept_)
    assert_allclose(dense_ridge.coef_, sparse_ridge.coef_, rtol=5e-07)
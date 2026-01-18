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
@pytest.mark.parametrize('with_sample_weight', [True, False])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_ridge_fit_intercept_sparse_sag(with_sample_weight, global_random_seed, csr_container):
    X, y = _make_sparse_offset_regression(n_features=5, n_samples=20, random_state=global_random_seed, X_offset=5.0)
    if with_sample_weight:
        rng = np.random.RandomState(global_random_seed)
        sample_weight = 1.0 + rng.uniform(size=X.shape[0])
    else:
        sample_weight = None
    X_csr = csr_container(X)
    params = dict(alpha=1.0, solver='sag', fit_intercept=True, tol=1e-10, max_iter=100000)
    dense_ridge = Ridge(**params)
    sparse_ridge = Ridge(**params)
    dense_ridge.fit(X, y, sample_weight=sample_weight)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        sparse_ridge.fit(X_csr, y, sample_weight=sample_weight)
    assert_allclose(dense_ridge.intercept_, sparse_ridge.intercept_, rtol=0.0001)
    assert_allclose(dense_ridge.coef_, sparse_ridge.coef_, rtol=0.0001)
    with pytest.warns(UserWarning, match='"sag" solver requires.*'):
        Ridge(solver='sag', fit_intercept=True, tol=0.001, max_iter=None).fit(X_csr, y)
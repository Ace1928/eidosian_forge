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
@pytest.mark.parametrize('n_samples,n_features', [[2, 3], [3, 2]])
@pytest.mark.parametrize('sparse_container', COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_sparse_design_with_sample_weights(n_samples, n_features, sparse_container):
    rng = np.random.RandomState(42)
    sparse_ridge = Ridge(alpha=1.0, fit_intercept=False)
    dense_ridge = Ridge(alpha=1.0, fit_intercept=False)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    sample_weights = rng.randn(n_samples) ** 2 + 1
    X_sparse = sparse_container(X)
    sparse_ridge.fit(X_sparse, y, sample_weight=sample_weights)
    dense_ridge.fit(X, y, sample_weight=sample_weights)
    assert_array_almost_equal(sparse_ridge.coef_, dense_ridge.coef_, decimal=6)
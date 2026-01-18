import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('sample_weight', [True, None])
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_scaler_without_centering(sample_weight, sparse_container):
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    X[:, 0] = 0.0
    X_sparse = sparse_container(X)
    if sample_weight:
        sample_weight = rng.rand(X.shape[0])
    with pytest.raises(ValueError):
        StandardScaler().fit(X_sparse)
    scaler = StandardScaler(with_mean=False).fit(X, sample_weight=sample_weight)
    X_scaled = scaler.transform(X, copy=True)
    assert not np.any(np.isnan(X_scaled))
    scaler_sparse = StandardScaler(with_mean=False).fit(X_sparse, sample_weight=sample_weight)
    X_sparse_scaled = scaler_sparse.transform(X_sparse, copy=True)
    assert not np.any(np.isnan(X_sparse_scaled.data))
    assert_array_almost_equal(scaler.mean_, scaler_sparse.mean_)
    assert_array_almost_equal(scaler.var_, scaler_sparse.var_)
    assert_array_almost_equal(scaler.scale_, scaler_sparse.scale_)
    assert_array_almost_equal(scaler.n_samples_seen_, scaler_sparse.n_samples_seen_)
    if sample_weight is None:
        assert_array_almost_equal(X_scaled.mean(axis=0), [0.0, -0.01, 2.24, -0.35, -0.78], 2)
        assert_array_almost_equal(X_scaled.std(axis=0), [0.0, 1.0, 1.0, 1.0, 1.0])
    X_sparse_scaled_mean, X_sparse_scaled_var = mean_variance_axis(X_sparse_scaled, 0)
    assert_array_almost_equal(X_sparse_scaled_mean, X_scaled.mean(axis=0))
    assert_array_almost_equal(X_sparse_scaled_var, X_scaled.var(axis=0))
    assert X_scaled is not X
    assert X_sparse_scaled is not X_sparse
    X_scaled_back = scaler.inverse_transform(X_scaled)
    assert X_scaled_back is not X
    assert X_scaled_back is not X_scaled
    assert_array_almost_equal(X_scaled_back, X)
    X_sparse_scaled_back = scaler_sparse.inverse_transform(X_sparse_scaled)
    assert X_sparse_scaled_back is not X_sparse
    assert X_sparse_scaled_back is not X_sparse_scaled
    assert_array_almost_equal(X_sparse_scaled_back.toarray(), X)
    if sparse_container in CSR_CONTAINERS:
        null_transform = StandardScaler(with_mean=False, with_std=False, copy=True)
        X_null = null_transform.fit_transform(X_sparse)
        assert_array_equal(X_null.data, X_sparse.data)
        X_orig = null_transform.inverse_transform(X_null)
        assert_array_equal(X_orig.data, X_sparse.data)
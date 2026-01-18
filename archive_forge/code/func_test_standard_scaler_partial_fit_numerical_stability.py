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
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_standard_scaler_partial_fit_numerical_stability(sparse_container):
    rng = np.random.RandomState(0)
    n_features = 2
    n_samples = 100
    offsets = rng.uniform(-1000000000000000.0, 1000000000000000.0, size=n_features)
    scales = rng.uniform(1000.0, 1000000.0, size=n_features)
    X = rng.randn(n_samples, n_features) * scales + offsets
    scaler_batch = StandardScaler().fit(X)
    scaler_incr = StandardScaler()
    for chunk in X:
        scaler_incr = scaler_incr.partial_fit(chunk.reshape(1, n_features))
    tol = 10 ** (-6)
    assert_allclose(scaler_incr.mean_, scaler_batch.mean_, rtol=tol)
    assert_allclose(scaler_incr.var_, scaler_batch.var_, rtol=tol)
    assert_allclose(scaler_incr.scale_, scaler_batch.scale_, rtol=tol)
    size = (100, 3)
    scale = 1e+20
    X = sparse_container(rng.randint(0, 2, size).astype(np.float64) * scale)
    scaler = StandardScaler(with_mean=False).fit(X)
    scaler_incr = StandardScaler(with_mean=False)
    for chunk in X:
        scaler_incr = scaler_incr.partial_fit(chunk)
    tol = 10 ** (-6)
    assert scaler.mean_ is not None
    assert_allclose(scaler_incr.var_, scaler.var_, rtol=tol)
    assert_allclose(scaler_incr.scale_, scaler.scale_, rtol=tol)
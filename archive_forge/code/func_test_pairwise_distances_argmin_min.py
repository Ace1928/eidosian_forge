import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('dok_container', DOK_CONTAINERS)
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_distances_argmin_min(dok_container, csr_container, global_dtype):
    X = np.asarray([[0], [1]], dtype=global_dtype)
    Y = np.asarray([[-2], [3]], dtype=global_dtype)
    Xsp = dok_container(X)
    Ysp = csr_container(Y, dtype=global_dtype)
    expected_idx = [0, 1]
    expected_vals = [2, 2]
    expected_vals_sq = [4, 4]
    idx, vals = pairwise_distances_argmin_min(X, Y, metric='euclidean')
    idx2 = pairwise_distances_argmin(X, Y, metric='euclidean')
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric='euclidean')
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric='euclidean')
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)
    assert type(idxsp) == np.ndarray
    assert type(valssp) == np.ndarray
    idx, vals = pairwise_distances_argmin_min(X, Y, metric='sqeuclidean')
    idx2, vals2 = pairwise_distances_argmin_min(X, Y, metric='euclidean', metric_kwargs={'squared': True})
    idx3 = pairwise_distances_argmin(X, Y, metric='sqeuclidean')
    idx4 = pairwise_distances_argmin(X, Y, metric='euclidean', metric_kwargs={'squared': True})
    assert_allclose(vals, expected_vals_sq)
    assert_allclose(vals2, expected_vals_sq)
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(idx3, expected_idx)
    assert_allclose(idx4, expected_idx)
    idx, vals = pairwise_distances_argmin_min(X, Y, metric='manhattan')
    idx2 = pairwise_distances_argmin(X, Y, metric='manhattan')
    assert_allclose(idx, expected_idx)
    assert_allclose(idx2, expected_idx)
    assert_allclose(vals, expected_vals)
    idxsp, valssp = pairwise_distances_argmin_min(Xsp, Ysp, metric='manhattan')
    idxsp2 = pairwise_distances_argmin(Xsp, Ysp, metric='manhattan')
    assert_allclose(idxsp, expected_idx)
    assert_allclose(idxsp2, expected_idx)
    assert_allclose(valssp, expected_vals)
    idx, vals = pairwise_distances_argmin_min(X, Y, metric=minkowski, metric_kwargs={'p': 2})
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)
    idx, vals = pairwise_distances_argmin_min(X, Y, metric='minkowski', metric_kwargs={'p': 2})
    assert_allclose(idx, expected_idx)
    assert_allclose(vals, expected_vals)
    rng = np.random.RandomState(0)
    X = rng.randn(97, 149)
    Y = rng.randn(111, 149)
    dist = pairwise_distances(X, Y, metric='manhattan')
    dist_orig_ind = dist.argmin(axis=0)
    dist_orig_val = dist[dist_orig_ind, range(len(dist_orig_ind))]
    dist_chunked_ind, dist_chunked_val = pairwise_distances_argmin_min(X, Y, axis=0, metric='manhattan')
    assert_allclose(dist_orig_ind, dist_chunked_ind, rtol=1e-07)
    assert_allclose(dist_orig_val, dist_chunked_val, rtol=1e-07)
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, Y, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(Y, X, axis=1)
    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)
    argmin_0, dist_0 = pairwise_distances_argmin_min(X, X, axis=0)
    argmin_1, dist_1 = pairwise_distances_argmin_min(X, X, axis=1)
    assert_allclose(dist_0, dist_1)
    assert_array_equal(argmin_0, argmin_1)
    argmin_0 = pairwise_distances_argmin(X, Y, axis=0)
    argmin_1 = pairwise_distances_argmin(Y, X, axis=1)
    assert_array_equal(argmin_0, argmin_1)
    argmin_0 = pairwise_distances_argmin(X, X, axis=0)
    argmin_1 = pairwise_distances_argmin(X, X, axis=1)
    assert_array_equal(argmin_0, argmin_1)
    argmin_C_contiguous = pairwise_distances_argmin(X, Y)
    argmin_F_contiguous = pairwise_distances_argmin(np.asfortranarray(X), np.asfortranarray(Y))
    assert_array_equal(argmin_C_contiguous, argmin_F_contiguous)
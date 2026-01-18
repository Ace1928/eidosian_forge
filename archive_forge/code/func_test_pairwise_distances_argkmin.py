import itertools
import re
import warnings
from functools import partial
import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('metric', CDIST_PAIRWISE_DISTANCES_REDUCTION_COMMON_METRICS)
@pytest.mark.parametrize('strategy', ('parallel_on_X', 'parallel_on_Y'))
@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_distances_argkmin(global_random_seed, metric, strategy, dtype, csr_container, n_queries=5, n_samples=100, k=10):
    rng = np.random.RandomState(global_random_seed)
    n_features = rng.choice([50, 500])
    translation = rng.choice([0, 1000000.0])
    spread = 1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread
    X_csr = csr_container(X)
    Y_csr = csr_container(Y)
    if metric == 'haversine':
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])
    metric_kwargs = _get_metric_params_list(metric, n_features)[0]
    if metric == 'euclidean':
        dist_matrix = euclidean_distances(X, Y)
    else:
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)
    argkmin_indices_ref = np.argsort(dist_matrix, axis=1)[:, :k]
    argkmin_distances_ref = np.zeros(argkmin_indices_ref.shape, dtype=np.float64)
    for row_idx in range(argkmin_indices_ref.shape[0]):
        argkmin_distances_ref[row_idx] = dist_matrix[row_idx, argkmin_indices_ref[row_idx]]
    for _X, _Y in itertools.product((X, X_csr), (Y, Y_csr)):
        argkmin_distances, argkmin_indices = ArgKmin.compute(_X, _Y, k, metric=metric, metric_kwargs=metric_kwargs, return_distance=True, chunk_size=n_samples // 4, strategy=strategy)
        ASSERT_RESULT[ArgKmin, dtype](argkmin_distances, argkmin_distances_ref, argkmin_indices, argkmin_indices_ref)
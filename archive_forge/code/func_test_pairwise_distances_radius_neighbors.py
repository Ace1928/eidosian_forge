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
def test_pairwise_distances_radius_neighbors(global_random_seed, metric, strategy, dtype, n_queries=5, n_samples=100):
    rng = np.random.RandomState(global_random_seed)
    n_features = rng.choice([50, 500])
    translation = rng.choice([0, 1000000.0])
    spread = 1000
    X = translation + rng.rand(n_queries, n_features).astype(dtype) * spread
    Y = translation + rng.rand(n_samples, n_features).astype(dtype) * spread
    metric_kwargs = _get_metric_params_list(metric, n_features, seed=global_random_seed)[0]
    if metric == 'euclidean':
        dist_matrix = euclidean_distances(X, Y)
    else:
        dist_matrix = cdist(X, Y, metric=metric, **metric_kwargs)
    radius = _non_trivial_radius(precomputed_dists=dist_matrix)
    neigh_indices_ref = []
    neigh_distances_ref = []
    for row in dist_matrix:
        ind = np.arange(row.shape[0])[row <= radius]
        dist = row[ind]
        sort = np.argsort(dist)
        ind, dist = (ind[sort], dist[sort])
        neigh_indices_ref.append(ind)
        neigh_distances_ref.append(dist)
    neigh_distances, neigh_indices = RadiusNeighbors.compute(X, Y, radius, metric=metric, metric_kwargs=metric_kwargs, return_distance=True, chunk_size=n_samples // 4, strategy=strategy, sort_results=True)
    ASSERT_RESULT[RadiusNeighbors, dtype](neigh_distances, neigh_distances_ref, neigh_indices, neigh_indices_ref, radius)
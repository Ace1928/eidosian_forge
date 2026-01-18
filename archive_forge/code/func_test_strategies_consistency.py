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
@pytest.mark.parametrize('Dispatcher', [ArgKmin, RadiusNeighbors])
def test_strategies_consistency(global_random_seed, global_dtype, Dispatcher, n_features=10):
    """Check that the results do not depend on the strategy used."""
    rng = np.random.RandomState(global_random_seed)
    metric = rng.choice(np.array(['euclidean', 'minkowski', 'manhattan', 'haversine'], dtype=object))
    n_samples_X, n_samples_Y = rng.choice([97, 100, 101, 500], size=2, replace=False)
    spread = 100
    X = rng.rand(n_samples_X, n_features).astype(global_dtype) * spread
    Y = rng.rand(n_samples_Y, n_features).astype(global_dtype) * spread
    if metric == 'haversine':
        X = np.ascontiguousarray(X[:, :2])
        Y = np.ascontiguousarray(Y[:, :2])
    if Dispatcher is ArgKmin:
        parameter = 10
        check_parameters = {}
        compute_parameters = {}
    else:
        radius = _non_trivial_radius(X=X, Y=Y, metric=metric)
        parameter = radius
        check_parameters = {'radius': radius}
        compute_parameters = {'sort_results': True}
    dist_par_X, indices_par_X = Dispatcher.compute(X, Y, parameter, metric=metric, metric_kwargs=_get_metric_params_list(metric, n_features, seed=global_random_seed)[0], chunk_size=n_samples_X // 4, strategy='parallel_on_X', return_distance=True, **compute_parameters)
    dist_par_Y, indices_par_Y = Dispatcher.compute(X, Y, parameter, metric=metric, metric_kwargs=_get_metric_params_list(metric, n_features, seed=global_random_seed)[0], chunk_size=n_samples_Y // 4, strategy='parallel_on_Y', return_distance=True, **compute_parameters)
    ASSERT_RESULT[Dispatcher, global_dtype](dist_par_X, dist_par_Y, indices_par_X, indices_par_Y, **check_parameters)
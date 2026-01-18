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
def test_radius_neighbors_factory_method_wrong_usages():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = 'euclidean'
    msg = 'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype=float32 and Y.dtype=float64'
    with pytest.raises(ValueError, match=msg):
        RadiusNeighbors.compute(X=X.astype(np.float32), Y=Y, radius=radius, metric=metric)
    msg = 'Only float64 or float32 datasets pairs are supported at this time, got: X.dtype=float64 and Y.dtype=int32'
    with pytest.raises(ValueError, match=msg):
        RadiusNeighbors.compute(X=X, Y=Y.astype(np.int32), radius=radius, metric=metric)
    with pytest.raises(ValueError, match='radius == -1.0, must be >= 0.'):
        RadiusNeighbors.compute(X=X, Y=Y, radius=-1, metric=metric)
    with pytest.raises(ValueError, match='Unrecognized metric'):
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric='wrong metric')
    with pytest.raises(ValueError, match='Buffer has wrong number of dimensions \\(expected 2, got 1\\)'):
        RadiusNeighbors.compute(X=np.array([1.0, 2.0]), Y=Y, radius=radius, metric=metric)
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        RadiusNeighbors.compute(X=np.asfortranarray(X), Y=Y, radius=radius, metric=metric)
    unused_metric_kwargs = {'p': 3}
    message = "Some metric_kwargs have been passed \\({'p': 3}\\) but"
    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=unused_metric_kwargs)
    metric_kwargs = {'p': 3, 'Y_norm_squared': sqeuclidean_row_norms(Y, num_threads=2)}
    message = "Some metric_kwargs have been passed \\({'p': 3, 'Y_norm_squared'"
    with pytest.warns(UserWarning, match=message):
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs)
    metric_kwargs = {'X_norm_squared': sqeuclidean_row_norms(X, num_threads=2), 'Y_norm_squared': sqeuclidean_row_norms(Y, num_threads=2)}
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=UserWarning)
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs)
    metric_kwargs = {'X_norm_squared': sqeuclidean_row_norms(X, num_threads=2)}
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=UserWarning)
        RadiusNeighbors.compute(X=X, Y=Y, radius=radius, metric=metric, metric_kwargs=metric_kwargs)
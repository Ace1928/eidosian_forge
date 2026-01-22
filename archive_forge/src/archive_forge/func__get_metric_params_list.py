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
def _get_metric_params_list(metric: str, n_features: int, seed: int=1):
    """Return list of dummy DistanceMetric kwargs for tests."""
    rng = np.random.RandomState(seed)
    if metric == 'minkowski':
        minkowski_kwargs = [dict(p=1.5), dict(p=2), dict(p=3), dict(p=np.inf), dict(p=3, w=rng.rand(n_features))]
        return minkowski_kwargs
    if metric == 'seuclidean':
        return [dict(V=rng.rand(n_features))]
    return [{}]
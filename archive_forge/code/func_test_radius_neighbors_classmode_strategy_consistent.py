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
@pytest.mark.parametrize('outlier_label', [None, 0, 3, 6, 9])
def test_radius_neighbors_classmode_strategy_consistent(outlier_label):
    rng = np.random.RandomState(1)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    radius = 5
    metric = 'manhattan'
    weights = 'uniform'
    Y_labels = rng.randint(low=0, high=10, size=100)
    unique_Y_labels = np.unique(Y_labels)
    results_X = RadiusNeighborsClassMode.compute(X=X, Y=Y, radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=outlier_label, strategy='parallel_on_X')
    results_Y = RadiusNeighborsClassMode.compute(X=X, Y=Y, radius=radius, metric=metric, weights=weights, Y_labels=Y_labels, unique_Y_labels=unique_Y_labels, outlier_label=outlier_label, strategy='parallel_on_Y')
    assert_allclose(results_X, results_Y)
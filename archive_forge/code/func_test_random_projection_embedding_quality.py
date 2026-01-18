import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_random_projection_embedding_quality(coo_container):
    data = make_sparse_random_data(coo_container, n_samples=8, n_features=5000, n_nonzeros=15000, random_state=0, sparse_format=None)
    eps = 0.2
    original_distances = euclidean_distances(data, squared=True)
    original_distances = original_distances.ravel()
    non_identical = original_distances != 0.0
    original_distances = original_distances[non_identical]
    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components='auto', eps=eps, random_state=0)
        projected = rp.fit_transform(data)
        projected_distances = euclidean_distances(projected, squared=True)
        projected_distances = projected_distances.ravel()
        projected_distances = projected_distances[non_identical]
        distances_ratio = projected_distances / original_distances
        assert distances_ratio.max() < 1 + eps
        assert 1 - eps < distances_ratio.min()
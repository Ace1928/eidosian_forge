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
def test_warning_n_components_greater_than_n_features(coo_container, global_random_seed):
    n_features = 20
    n_samples = 5
    n_nonzeros = int(n_features / 4)
    data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=global_random_seed, sparse_format=None)
    for RandomProjection in all_RandomProjection:
        with pytest.warns(DataDimensionalityWarning):
            RandomProjection(n_components=n_features + 1).fit(data)
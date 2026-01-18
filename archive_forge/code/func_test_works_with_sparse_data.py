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
def test_works_with_sparse_data(coo_container, global_random_seed):
    n_features = 20
    n_samples = 5
    n_nonzeros = int(n_features / 4)
    dense_data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=global_random_seed, sparse_format=None)
    sparse_data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=global_random_seed, sparse_format='csr')
    for RandomProjection in all_RandomProjection:
        rp_dense = RandomProjection(n_components=3, random_state=1).fit(dense_data)
        rp_sparse = RandomProjection(n_components=3, random_state=1).fit(sparse_data)
        assert_array_almost_equal(densify(rp_dense.components_), densify(rp_sparse.components_))
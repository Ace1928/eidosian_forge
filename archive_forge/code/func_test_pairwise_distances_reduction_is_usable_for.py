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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_distances_reduction_is_usable_for(csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(100, 10)
    Y = rng.rand(100, 10)
    X_csr = csr_container(X)
    Y_csr = csr_container(Y)
    metric = 'manhattan'
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y_csr, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y_csr, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X.astype(np.float64), Y.astype(np.float64), metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X.astype(np.float32), Y.astype(np.float32), metric)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X.astype(np.int64), Y.astype(np.int64), metric)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X, Y, metric='pyfunc')
    assert not BaseDistancesReductionDispatcher.is_usable_for(X.astype(np.float32), Y, metric)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X, Y.astype(np.int32), metric)
    assert not BaseDistancesReductionDispatcher.is_usable_for(np.asfortranarray(X), Y, metric)
    assert BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y, metric='euclidean')
    assert BaseDistancesReductionDispatcher.is_usable_for(X, Y_csr, metric='sqeuclidean')
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y_csr, metric='sqeuclidean')
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr, Y_csr, metric='euclidean')
    X_csr_0_nnz = csr_container(X * 0)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_0_nnz, Y, metric)
    X_csr_int64 = csr_container(X)
    X_csr_int64.indices = X_csr_int64.indices.astype(np.int64)
    assert not BaseDistancesReductionDispatcher.is_usable_for(X_csr_int64, Y, metric)
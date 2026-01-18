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
@pytest.mark.parametrize('random_projection_cls', all_RandomProjection)
def test_random_projection_numerical_consistency(random_projection_cls):
    atol = 1e-05
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp_32 = random_projection_cls(random_state=0)
    rp_64 = random_projection_cls(random_state=0)
    projection_32 = rp_32.fit_transform(X.astype(np.float32))
    projection_64 = rp_64.fit_transform(X.astype(np.float64))
    assert_allclose(projection_64, projection_32, atol=atol)
    assert_allclose_dense_sparse(rp_32.components_, rp_64.components_)
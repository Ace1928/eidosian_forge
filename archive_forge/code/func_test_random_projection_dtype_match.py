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
@pytest.mark.parametrize('input_dtype, expected_dtype', ((np.float32, np.float32), (np.float64, np.float64), (np.int32, np.float64), (np.int64, np.float64)))
def test_random_projection_dtype_match(random_projection_cls, input_dtype, expected_dtype):
    rng = np.random.RandomState(42)
    X = rng.rand(25, 3000)
    rp = random_projection_cls(random_state=0)
    transformed = rp.fit_transform(X.astype(input_dtype))
    assert rp.components_.dtype == expected_dtype
    assert transformed.dtype == expected_dtype
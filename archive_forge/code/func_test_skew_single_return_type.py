from __future__ import annotations
import warnings
import pytest
from packaging.version import parse as parse_version
import numpy as np
import dask.array as da
import dask.array.stats
from dask.array.utils import allclose, assert_eq
from dask.delayed import Delayed
def test_skew_single_return_type():
    """This function tests the return type for the skew method for a 1d array."""
    numpy_array = np.random.random(size=(30,))
    dask_array = da.from_array(numpy_array, 3)
    result = dask.array.stats.skew(dask_array).compute()
    assert isinstance(result, np.float64)
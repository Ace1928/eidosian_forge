from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_ravel_multi_index_unknown_shape_fails():
    multi_index1 = da.from_array([2, -1, 3, -1], chunks=2)
    multi_index1 = multi_index1[multi_index1 > 0]
    multi_index2 = da.from_array([[1, 2], [-1, -1], [3, 4], [5, 6], [7, 8], [-1, -1]], chunks=(2, 1))
    multi_index2 = multi_index2[(multi_index2 > 0).all(axis=1)]
    multi_index = [1, multi_index1, multi_index2]
    assert np.isnan(multi_index1.shape).any()
    assert np.isnan(multi_index2.shape).any()
    with pytest.raises(ValueError, match="Arrays' chunk sizes"):
        da.ravel_multi_index(multi_index, dims=(8, 9, 10))
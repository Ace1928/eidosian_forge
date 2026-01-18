from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_overlap_internal_asymmetric():
    x = np.arange(64).reshape((8, 8))
    d = da.from_array(x, chunks=(4, 4))
    result = overlap_internal(d, {0: (2, 0), 1: (1, 0)})
    assert result.chunks == ((4, 6), (4, 5))
    expected = np.array([[0, 1, 2, 3, 3, 4, 5, 6, 7], [8, 9, 10, 11, 11, 12, 13, 14, 15], [16, 17, 18, 19, 19, 20, 21, 22, 23], [24, 25, 26, 27, 27, 28, 29, 30, 31], [16, 17, 18, 19, 19, 20, 21, 22, 23], [24, 25, 26, 27, 27, 28, 29, 30, 31], [32, 33, 34, 35, 35, 36, 37, 38, 39], [40, 41, 42, 43, 43, 44, 45, 46, 47], [48, 49, 50, 51, 51, 52, 53, 54, 55], [56, 57, 58, 59, 59, 60, 61, 62, 63]])
    assert_eq(result, expected)
    assert same_keys(overlap_internal(d, {0: (2, 0), 1: (1, 0)}), result)
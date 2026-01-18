from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_overlap_internal_asymmetric_small():
    x = np.arange(32).reshape((2, 16))
    d = da.from_array(x, chunks=(2, 4))
    result = overlap_internal(d, {0: (0, 0), 1: (1, 1)})
    assert result.chunks == ((2,), (5, 6, 6, 5))
    expected = np.array([[0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10, 11, 12, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 19, 20, 21, 22, 23, 24, 23, 24, 25, 26, 27, 28, 27, 28, 29, 30, 31]])
    assert_eq(result, expected)
    assert same_keys(overlap_internal(d, {0: (0, 0), 1: (1, 1)}), result)
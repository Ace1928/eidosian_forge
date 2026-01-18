from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_none_boundaries():
    x = da.from_array(np.arange(16).reshape(4, 4), chunks=(2, 2))
    exp = boundaries(x, 2, {0: 'none', 1: 33})
    res = np.array([[33, 33, 0, 1, 2, 3, 33, 33], [33, 33, 4, 5, 6, 7, 33, 33], [33, 33, 8, 9, 10, 11, 33, 33], [33, 33, 12, 13, 14, 15, 33, 33]])
    assert_eq(exp, res)
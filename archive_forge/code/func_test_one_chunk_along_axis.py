from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_one_chunk_along_axis():
    a = np.arange(2 * 9).reshape(2, 9)
    darr = da.from_array(a, chunks=((2,), (2, 2, 2, 3)))
    g = overlap(darr, depth=0, boundary=0)
    assert a.shape == g.shape
from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_overlap_small():
    x = da.ones((10, 10), chunks=(5, 5))
    y = x.map_overlap(lambda x: x, depth=1, boundary='none')
    assert len(y.dask) < 200
    y = x.map_overlap(lambda x: x, depth=1, boundary='none')
    assert len(y.dask) < 100
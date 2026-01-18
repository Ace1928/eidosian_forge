from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('window_shape, axis', [((10,), 0), ((2,), 3), (-1, 0), (2, (0, 1)), (2, None), (0, None)])
def test_sliding_window_errors(window_shape, axis):
    arr = da.zeros((4, 3))
    with pytest.raises(ValueError):
        sliding_window_view(arr, window_shape, axis)
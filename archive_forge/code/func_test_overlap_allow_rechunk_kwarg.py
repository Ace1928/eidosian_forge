from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_overlap_allow_rechunk_kwarg():
    arr = da.arange(6, chunks=5)
    da.overlap.overlap(arr, 2, 'reflect', allow_rechunk=True)
    arr.map_overlap(lambda x: x, 2, 'reflect', allow_rechunk=True)
    with pytest.raises(ValueError):
        da.overlap.overlap(arr, 2, 'reflect', allow_rechunk=False)
    with pytest.raises(ValueError):
        arr.map_overlap(lambda x: x, 2, 'reflect', allow_rechunk=False)
    arr = da.arange(6, chunks=4)
    da.overlap.overlap(arr, 2, 'reflect', allow_rechunk=False)
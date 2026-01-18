from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_slice_broadcast(dtype) -> None:
    arr = xr.DataArray(['aafootwo', 'aabartwo', 'aabazqux']).astype(dtype)
    start = xr.DataArray([1, 2, 3])
    stop = 5
    result = arr.str.slice(start=start, stop=stop)
    exp = xr.DataArray(['afoo', 'bar', 'az']).astype(dtype)
    assert result.dtype == exp.dtype
    assert_equal(result, exp)
from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_slice_replace_broadcast(dtype) -> None:
    values = xr.DataArray(['short', 'a bit longer', 'evenlongerthanthat', '']).astype(dtype)
    start = 2
    stop = np.array([4, 5, None, 7])
    repl = 'test'
    expected = xr.DataArray(['shtestt', 'a test longer', 'evtest', 'test']).astype(dtype)
    result = values.str.slice_replace(start, stop, repl)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
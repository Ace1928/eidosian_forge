from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_pad_center_ljust_rjust(dtype) -> None:
    values = xr.DataArray(['a', 'b', 'c', 'eeeee']).astype(dtype)
    result = values.str.center(5)
    expected = xr.DataArray(['  a  ', '  b  ', '  c  ', 'eeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.pad(5, side='both')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.ljust(5)
    expected = xr.DataArray(['a    ', 'b    ', 'c    ', 'eeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.pad(5, side='right')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.rjust(5)
    expected = xr.DataArray(['    a', '    b', '    c', 'eeeee']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.pad(5, side='left')
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
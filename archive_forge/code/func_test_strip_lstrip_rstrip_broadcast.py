from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_strip_lstrip_rstrip_broadcast(dtype) -> None:
    values = xr.DataArray(['xxABCxx', 'yy BNSD', 'LDFJH zz']).astype(dtype)
    to_strip = xr.DataArray(['x', 'y', 'z']).astype(dtype)
    result = values.str.strip(to_strip)
    expected = xr.DataArray(['ABC', ' BNSD', 'LDFJH ']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.lstrip(to_strip)
    expected = xr.DataArray(['ABCxx', ' BNSD', 'LDFJH zz']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.rstrip(to_strip)
    expected = xr.DataArray(['xxABC', 'yy BNSD', 'LDFJH ']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
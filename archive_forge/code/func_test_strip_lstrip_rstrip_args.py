from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_strip_lstrip_rstrip_args(dtype) -> None:
    values = xr.DataArray(['xxABCxx', 'xx BNSD', 'LDFJH xx']).astype(dtype)
    result = values.str.strip('x')
    expected = xr.DataArray(['ABC', ' BNSD', 'LDFJH ']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.lstrip('x')
    expected = xr.DataArray(['ABCxx', ' BNSD', 'LDFJH xx']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.rstrip('x')
    expected = xr.DataArray(['xxABC', 'xx BNSD', 'LDFJH ']).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
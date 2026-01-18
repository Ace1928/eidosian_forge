from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_starts_ends_with(dtype) -> None:
    values = xr.DataArray(['om', 'foo_nom', 'nom', 'bar_foo', 'foo']).astype(dtype)
    result = values.str.startswith('foo')
    expected = xr.DataArray([False, True, False, False, True])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    result = values.str.endswith('foo')
    expected = xr.DataArray([False, False, False, True, True])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
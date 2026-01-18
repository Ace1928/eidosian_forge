from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_repeat_broadcast(dtype) -> None:
    values = xr.DataArray(['a', 'b', 'c', 'd'], dims=['X']).astype(dtype)
    reps = xr.DataArray([3, 4], dims=['Y'])
    result = values.str.repeat(reps)
    result_mul = values.str * reps
    expected = xr.DataArray([['aaa', 'aaaa'], ['bbb', 'bbbb'], ['ccc', 'cccc'], ['ddd', 'dddd']], dims=['X', 'Y']).astype(dtype)
    assert result.dtype == expected.dtype
    assert result_mul.dtype == expected.dtype
    assert_equal(result_mul, expected)
    assert_equal(result, expected)
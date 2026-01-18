from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 4), ([0, 2], 4), ([1, 0], 0), ([0, 0], 0)))
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_sum_nan(weights, expected, skipna):
    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum(skipna=skipna)
    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)
    assert_equal(expected, result)
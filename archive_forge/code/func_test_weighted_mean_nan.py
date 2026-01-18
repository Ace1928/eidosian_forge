from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([4, 6], 2.0), ([1, 0], np.nan), ([0, 0], np.nan)))
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_mean_nan(weights, expected, skipna):
    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)
    result = da.weighted(weights).mean(skipna=skipna)
    assert_equal(expected, result)
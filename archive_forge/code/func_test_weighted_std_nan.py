from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([4, 6], 0), ([1, 0], np.nan), ([0, 0], np.nan)))
def test_weighted_std_nan(weights, expected):
    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    expected = DataArray(expected)
    result = da.weighted(weights).std()
    assert_equal(expected, result)
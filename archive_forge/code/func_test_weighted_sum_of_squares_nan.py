from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 0), ([2, 0], 0), ([0, 0], 0), ([-1, 1], 0)))
def test_weighted_sum_of_squares_nan(weights, expected):
    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum_of_squares()
    expected = DataArray(expected)
    assert_equal(expected, result)
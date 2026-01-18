from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 5), ([0, 2], 4), ([0, 0], 0)))
def test_weighted_sum_no_nan(weights, expected):
    da = DataArray([1, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum()
    expected = DataArray(expected)
    assert_equal(expected, result)
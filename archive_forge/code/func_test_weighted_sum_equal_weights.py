from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('da', ([1.0, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize('factor', [0, 1, 3.14])
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_sum_equal_weights(da, factor, skipna):
    da = DataArray(da)
    weights = xr.full_like(da, factor)
    expected = da.sum(skipna=skipna) * factor
    result = da.weighted(weights).sum(skipna=skipna)
    assert_equal(expected, result)
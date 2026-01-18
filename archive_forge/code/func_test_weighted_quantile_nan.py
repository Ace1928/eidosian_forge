from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_quantile_nan(skipna):
    da = DataArray([0, 1, 2, 3, np.nan])
    w = DataArray([1, 0, 1, 0, 1])
    q = [0.5, 0.75]
    result = da.weighted(w).quantile(q, skipna=skipna)
    if skipna:
        expected = DataArray(np.quantile([0, 2], q), coords={'quantile': q})
    else:
        expected = DataArray(np.full(len(q), np.nan), coords={'quantile': q})
    assert_allclose(expected, result)
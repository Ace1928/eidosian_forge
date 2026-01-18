from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
def test_weighted_quantile_simple():
    da = DataArray([0, 1, 2, 3])
    w = DataArray([1, 0, 1, 0])
    w_eps = DataArray([1, 0.0001, 1, 0.0001])
    q = 0.75
    expected = DataArray(np.quantile([0, 2], q), coords={'quantile': q})
    assert_equal(expected, da.weighted(w).quantile(q))
    assert_allclose(expected, da.weighted(w_eps).quantile(q), rtol=0.001)
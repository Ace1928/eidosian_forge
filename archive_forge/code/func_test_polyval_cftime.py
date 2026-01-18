from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_cftime
@pytest.mark.parametrize('use_dask', [pytest.param(False, id='nodask'), pytest.param(True, id='dask')])
@pytest.mark.parametrize('date', ['1970-01-01', '0753-04-21'])
def test_polyval_cftime(use_dask: bool, date: str) -> None:
    import cftime
    x = xr.DataArray(xr.date_range(date, freq='1s', periods=3, use_cftime=True), dims='x')
    coeffs = xr.DataArray([0, 1], dims='degree', coords={'degree': [0, 1]})
    if use_dask:
        if not has_dask:
            pytest.skip('requires dask')
        coeffs = coeffs.chunk({'degree': 2})
        x = x.chunk({'x': 2})
    with raise_if_dask_computes(max_computes=1):
        actual = xr.polyval(coord=x, coeffs=coeffs)
    t0 = xr.date_range(date, periods=1)[0]
    offset = (t0 - cftime.DatetimeGregorian(1970, 1, 1)).total_seconds() * 1000000000.0
    expected = xr.DataArray([0, 1000000000.0, 2000000000.0], dims='x', coords={'x': xr.date_range(date, freq='1s', periods=3, use_cftime=True)}) + offset
    xr.testing.assert_allclose(actual, expected)
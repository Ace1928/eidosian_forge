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
@pytest.mark.parametrize('use_dask', [pytest.param(False, id='nodask'), pytest.param(True, id='dask')])
@pytest.mark.parametrize('x', [pytest.param(xr.DataArray([0, 1, 2], dims='x'), id='simple'), pytest.param(xr.DataArray(pd.date_range('1970-01-01', freq='ns', periods=3), dims='x'), id='datetime'), pytest.param(xr.DataArray(np.array([0, 1, 2], dtype='timedelta64[ns]'), dims='x'), id='timedelta')])
@pytest.mark.parametrize('y', [pytest.param(xr.DataArray([1, 6, 17], dims='x'), id='1D'), pytest.param(xr.DataArray([[1, 6, 17], [34, 57, 86]], dims=('y', 'x')), id='2D')])
def test_polyfit_polyval_integration(use_dask: bool, x: xr.DataArray, y: xr.DataArray) -> None:
    y.coords['x'] = x
    if use_dask:
        if not has_dask:
            pytest.skip('requires dask')
        y = y.chunk({'x': 2})
    fit = y.polyfit(dim='x', deg=2)
    evaluated = xr.polyval(y.x, fit.polyfit_coefficients)
    expected = y.transpose(*evaluated.dims)
    xr.testing.assert_allclose(evaluated.variable, expected.variable)
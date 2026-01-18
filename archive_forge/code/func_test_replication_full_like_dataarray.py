from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), pytest.param('coords', marks=pytest.mark.xfail(reason="can't copy quantity into non-quantity"))))
def test_replication_full_like_dataarray(variant, dtype):
    unit = unit_registry.m
    variants = {'data': (unit, 1, 1), 'dims': (1, unit, 1), 'coords': (1, 1, unit)}
    data_unit, dim_unit, coord_unit = variants.get(variant)
    array = np.linspace(0, 5, 10) * data_unit
    x = np.arange(10) * dim_unit
    u = np.linspace(0, 1, 10) * coord_unit
    data_array = xr.DataArray(data=array, dims='x', coords={'x': x, 'u': ('x', u)})
    fill_value = -1 * unit_registry.degK
    units = extract_units(data_array)
    units[data_array.name] = fill_value.units
    expected = attach_units(xr.full_like(strip_units(data_array), fill_value=strip_units(fill_value)), units)
    actual = xr.full_like(data_array, fill_value=fill_value)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)
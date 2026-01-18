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
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units")), 'coords'))
@pytest.mark.parametrize('func', (xr.zeros_like, xr.ones_like))
def test_replication_dataset(func, variant, dtype):
    unit = unit_registry.m
    variants = {'data': ((unit_registry.m, unit_registry.Pa), 1, 1), 'dims': ((1, 1), unit, 1), 'coords': ((1, 1), 1, unit)}
    (data_unit1, data_unit2), dim_unit, coord_unit = variants.get(variant)
    array1 = np.linspace(0, 10, 20).astype(dtype) * data_unit1
    array2 = np.linspace(5, 10, 10).astype(dtype) * data_unit2
    x = np.arange(20).astype(dtype) * dim_unit
    y = np.arange(10).astype(dtype) * dim_unit
    u = np.linspace(0, 1, 10) * coord_unit
    ds = xr.Dataset(data_vars={'a': ('x', array1), 'b': ('y', array2)}, coords={'x': x, 'y': y, 'u': ('y', u)})
    units = {name: unit for name, unit in extract_units(ds).items() if name not in ds.data_vars}
    expected = attach_units(func(strip_units(ds)), units)
    actual = func(ds)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)
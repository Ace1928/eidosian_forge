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
@pytest.mark.parametrize('func', (method('unstack'), method('reset_index', 'v'), method('reorder_levels')), ids=repr)
@pytest.mark.parametrize('variant', ('data', pytest.param('dims', marks=pytest.mark.skip(reason="indexes don't support units"))))
def test_stacking_stacked(self, variant, func, dtype):
    variants = {'data': (unit_registry.m, 1), 'dims': (1, unit_registry.m)}
    data_unit, dim_unit = variants.get(variant)
    array1 = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * data_unit
    array2 = np.linspace(-10, 0, 5 * 10 * 15).reshape(5, 10, 15).astype(dtype) * data_unit
    x = np.arange(array1.shape[0]) * dim_unit
    y = np.arange(array1.shape[1]) * dim_unit
    z = np.arange(array2.shape[2]) * dim_unit
    ds = xr.Dataset(data_vars={'a': (('x', 'y'), array1), 'b': (('x', 'y', 'z'), array2)}, coords={'x': x, 'y': y, 'z': z})
    units = extract_units(ds)
    stacked = ds.stack(v=('x', 'y'))
    expected = attach_units(func(strip_units(stacked)), units)
    actual = func(stacked)
    assert_units_equal(expected, actual)
    if func.name == 'reset_index':
        assert_equal(expected, actual, check_default_indexes=False)
    else:
        assert_equal(expected, actual)
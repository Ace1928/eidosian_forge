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
@pytest.mark.parametrize('func', (method('transpose', 'y', 'x', 'z1', 'z2'), method('stack', u=('x', 'y')), method('set_index', x='x2'), method('shift', x=2), pytest.param(method('rank', dim='x'), marks=pytest.mark.skip(reason='rank not implemented for non-ndarray')), method('roll', x=2, roll_coords=False), method('sortby', 'x2')), ids=repr)
def test_stacking_reordering(self, func, dtype):
    array1 = np.linspace(0, 10, 2 * 5 * 10).reshape(2, 5, 10).astype(dtype) * unit_registry.Pa
    array2 = np.linspace(0, 10, 2 * 5 * 15).reshape(2, 5, 15).astype(dtype) * unit_registry.degK
    x = np.arange(array1.shape[0])
    y = np.arange(array1.shape[1])
    z1 = np.arange(array1.shape[2])
    z2 = np.arange(array2.shape[2])
    x2 = np.linspace(0, 1, array1.shape[0])[::-1]
    ds = xr.Dataset(data_vars={'a': (('x', 'y', 'z1'), array1), 'b': (('x', 'y', 'z2'), array2)}, coords={'x': x, 'y': y, 'z1': z1, 'z2': z2, 'x2': ('x', x2)})
    units = extract_units(ds)
    expected = attach_units(func(strip_units(ds)), units)
    actual = func(ds)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)
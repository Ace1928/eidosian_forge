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
@pytest.mark.xfail(reason="stacked dimension's labels have to be hashable, but is a numpy.array")
def test_to_stacked_array(self, dtype):
    labels = range(5) * unit_registry.s
    arrays = {name: np.linspace(0, 1, 10).astype(dtype) * unit_registry.m for name in labels}
    ds = xr.Dataset({name: ('x', array) for name, array in arrays.items()})
    units = {None: unit_registry.m, 'y': unit_registry.s}
    func = method('to_stacked_array', 'z', variable_dim='y', sample_dims=['x'])
    actual = func(ds).rename(None)
    expected = attach_units(func(strip_units(ds)).rename(None), units)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)
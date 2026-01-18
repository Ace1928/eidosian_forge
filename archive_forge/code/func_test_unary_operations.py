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
@pytest.mark.parametrize('func', (pytest.param(operator.neg, id='negate'), pytest.param(abs, id='absolute'), pytest.param(np.round, id='round')))
def test_unary_operations(self, func, dtype):
    array = np.arange(10).astype(dtype) * unit_registry.m
    data_array = xr.DataArray(data=array)
    units = extract_units(func(array))
    expected = attach_units(func(strip_units(data_array)), units)
    actual = func(data_array)
    assert_units_equal(expected, actual)
    assert_identical(expected, actual)
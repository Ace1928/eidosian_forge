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
def test_pad(self, dtype):
    a = np.linspace(0, 5, 10).astype(dtype) * unit_registry.Pa
    b = np.linspace(-5, 0, 10).astype(dtype) * unit_registry.degK
    ds = xr.Dataset({'a': ('x', a), 'b': ('x', b)})
    units = extract_units(ds)
    expected = attach_units(strip_units(ds).pad(x=(2, 3)), units)
    actual = ds.pad(x=(2, 3))
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)
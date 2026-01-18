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
@pytest.mark.parametrize('func', (method('astype', float), method('conj'), method('argsort'), method('conjugate'), method('round')), ids=repr)
def test_numpy_methods(self, func, dtype):
    a = np.linspace(1, -1, 10) * unit_registry.Pa
    b = np.linspace(-1, 1, 15) * unit_registry.degK
    ds = xr.Dataset({'a': ('x', a), 'b': ('y', b)})
    units_a = array_extract_units(func(a))
    units_b = array_extract_units(func(b))
    units = {'a': units_a, 'b': units_b}
    actual = func(ds)
    expected = attach_units(func(strip_units(ds)), units)
    assert_units_equal(expected, actual)
    assert_equal(expected, actual)
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
def test_output_wrong_dim_size() -> None:
    array = np.arange(10)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    def truncate(array):
        return array[:5]

    def apply_truncate_broadcast_invalid(obj):
        return apply_ufunc(truncate, obj)
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_broadcast_invalid(variable)
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_broadcast_invalid(data_array)
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_broadcast_invalid(dataset)

    def apply_truncate_x_x_invalid(obj):
        return apply_ufunc(truncate, obj, input_core_dims=[['x']], output_core_dims=[['x']])
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_x_x_invalid(variable)
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_x_x_invalid(data_array)
    with pytest.raises(ValueError, match='size of dimension'):
        apply_truncate_x_x_invalid(dataset)

    def apply_truncate_x_z(obj):
        return apply_ufunc(truncate, obj, input_core_dims=[['x']], output_core_dims=[['z']])
    assert_identical(xr.Variable('z', array[:5]), apply_truncate_x_z(variable))
    assert_identical(xr.DataArray(array[:5], dims=['z']), apply_truncate_x_z(data_array))
    assert_identical(xr.Dataset({'y': ('z', array[:5])}), apply_truncate_x_z(dataset))

    def apply_truncate_x_x_valid(obj):
        return apply_ufunc(truncate, obj, input_core_dims=[['x']], output_core_dims=[['x']], exclude_dims={'x'})
    assert_identical(xr.Variable('x', array[:5]), apply_truncate_x_x_valid(variable))
    assert_identical(xr.DataArray(array[:5], dims=['x']), apply_truncate_x_x_valid(data_array))
    assert_identical(xr.Dataset({'y': ('x', array[:5])}), apply_truncate_x_x_valid(dataset))
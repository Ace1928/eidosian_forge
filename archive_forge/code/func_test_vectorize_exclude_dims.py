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
def test_vectorize_exclude_dims() -> None:
    data_array_a = xr.DataArray([[0, 1, 2], [1, 2, 3]], dims=('x', 'y'))
    data_array_b = xr.DataArray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dims=('x', 'y'))
    expected = xr.DataArray([3, 5], dims=['x'])
    actual = apply_ufunc(pandas_median_add, data_array_a, data_array_b, input_core_dims=[['y'], ['y']], vectorize=True, exclude_dims=set('y'))
    assert_identical(expected, actual)
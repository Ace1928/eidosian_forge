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
@requires_dask
@pytest.mark.parametrize('data_array', [xr.DataArray([[0, 1, 2], [1, 2, 3]], dims=('x', 'y')), xr.DataArray([[0 + 0j, 1 + 2j, 2 + 1j]], dims=('x', 'y'))])
def test_vectorize_dask_dtype_without_output_dtypes(data_array) -> None:
    expected = data_array.copy()
    actual = apply_ufunc(identity, data_array.chunk({'x': 1}), vectorize=True, dask='parallelized')
    assert_identical(expected, actual)
    assert expected.dtype == actual.dtype
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
def test_vectorize_dask_new_output_dims() -> None:
    data_array = xr.DataArray([[0, 1, 2], [1, 2, 3]], dims=('x', 'y'))
    func = lambda x: x[np.newaxis, ...]
    expected = data_array.expand_dims('z')
    actual = apply_ufunc(func, data_array.chunk({'x': 1}), output_core_dims=[['z']], vectorize=True, dask='parallelized', output_dtypes=[float], dask_gufunc_kwargs=dict(output_sizes={'z': 1})).transpose(*expected.dims)
    assert_identical(expected, actual)
    with pytest.raises(ValueError, match="dimension 'z1' in 'output_sizes' must correspond"):
        apply_ufunc(func, data_array.chunk({'x': 1}), output_core_dims=[['z']], vectorize=True, dask='parallelized', output_dtypes=[float], dask_gufunc_kwargs=dict(output_sizes={'z1': 1}))
    with pytest.raises(ValueError, match="dimension 'z' in 'output_core_dims' needs corresponding"):
        apply_ufunc(func, data_array.chunk({'x': 1}), output_core_dims=[['z']], vectorize=True, dask='parallelized', output_dtypes=[float])
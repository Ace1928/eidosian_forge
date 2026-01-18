from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('dim', ('a', 'b', 'c', ('a', 'b'), ('a', 'b', 'c'), None))
@pytest.mark.parametrize('q', (0.5, (0.1, 0.9), (0.2, 0.4, 0.6, 0.8)))
@pytest.mark.parametrize('add_nans', (True, False))
@pytest.mark.parametrize('skipna', (None, True, False))
def test_weighted_quantile_3D(dim, q, add_nans, skipna):
    dims = ('a', 'b', 'c')
    coords = dict(a=[0, 1, 2], b=[0, 1, 2, 3], c=[0, 1, 2, 3, 4])
    data = np.arange(60).reshape(3, 4, 5).astype(float)
    if add_nans:
        c = int(data.size * 0.25)
        data.ravel()[np.random.choice(data.size, c, replace=False)] = np.nan
    da = DataArray(data, dims=dims, coords=coords)
    weights = xr.ones_like(da)
    result = da.weighted(weights).quantile(q, dim=dim, skipna=skipna)
    expected = da.quantile(q, dim=dim, skipna=skipna)
    assert_allclose(expected, result)
    ds = da.to_dataset(name='data')
    result2 = ds.weighted(weights).quantile(q, dim=dim, skipna=skipna)
    assert_allclose(expected, result2.data)
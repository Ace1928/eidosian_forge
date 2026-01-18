from __future__ import annotations
from collections.abc import Iterable
from typing import Any
import numpy as np
import pytest
import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import (
@pytest.mark.parametrize('operation', ('sum_of_weights', 'sum', 'mean', 'quantile'))
@pytest.mark.parametrize('as_dataset', (True, False))
def test_weighted_bad_dim(operation, as_dataset):
    data = DataArray(np.random.randn(2, 2))
    weights = xr.ones_like(data)
    if as_dataset:
        data = data.to_dataset(name='data')
    kwargs = {'dim': 'bad_dim'}
    if operation == 'quantile':
        kwargs['q'] = 0.5
    with pytest.raises(ValueError, match=f"Dimensions \\('bad_dim',\\) not found in {data.__class__.__name__}Weighted dimensions \\(('dim_0', 'dim_1'|'dim_1', 'dim_0')\\)"):
        getattr(data.weighted(weights), operation)(**kwargs)
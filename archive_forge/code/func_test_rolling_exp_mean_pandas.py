from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('dim', ['time', 'x'])
@pytest.mark.parametrize('window_type, window', [['span', 5], ['alpha', 0.5], ['com', 0.5], ['halflife', 5]])
@pytest.mark.parametrize('backend', ['numpy'], indirect=True)
def test_rolling_exp_mean_pandas(self, da, dim, window_type, window) -> None:
    da = da.isel(a=0).where(lambda x: x > 0.2)
    result = da.rolling_exp(window_type=window_type, **{dim: window}).mean()
    assert isinstance(result, DataArray)
    pandas_array = da.to_pandas()
    assert pandas_array.index.name == 'time'
    if dim == 'x':
        pandas_array = pandas_array.T
    expected = xr.DataArray(pandas_array.ewm(**{window_type: window}).mean()).transpose(*da.dims)
    assert_allclose(expected.variable, result.variable)
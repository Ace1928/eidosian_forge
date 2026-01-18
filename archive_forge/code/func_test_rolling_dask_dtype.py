from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('dtype', ['int', 'float32', 'float64'])
def test_rolling_dask_dtype(self, dtype) -> None:
    data = DataArray(np.array([1, 2, 3], dtype=dtype), dims='x', coords={'x': [1, 2, 3]})
    unchunked_result = data.rolling(x=3, min_periods=1).mean()
    chunked_result = data.chunk({'x': 1}).rolling(x=3, min_periods=1).mean()
    assert chunked_result.dtype == unchunked_result.dtype
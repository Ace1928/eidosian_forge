from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@requires_dask
@pytest.mark.parametrize('name', ('mean', 'count'))
@pytest.mark.parametrize('center', (True, False, None))
@pytest.mark.parametrize('min_periods', (1, None))
@pytest.mark.parametrize('window', (7, 8))
@pytest.mark.parametrize('backend', ['dask'], indirect=True)
def test_rolling_wrapped_dask(self, da, name, center, min_periods, window) -> None:
    rolling_obj = da.rolling(time=window, min_periods=min_periods, center=center)
    actual = getattr(rolling_obj, name)().load()
    if name != 'count':
        with pytest.warns(DeprecationWarning, match='Reductions are applied'):
            getattr(rolling_obj, name)(dim='time')
    rolling_obj = da.load().rolling(time=window, min_periods=min_periods, center=center)
    expected = getattr(rolling_obj, name)()
    assert_allclose(actual, expected)
    rolling_obj = da.chunk().rolling(time=window, min_periods=min_periods, center=center)
    actual = getattr(rolling_obj, name)().load()
    assert_allclose(actual, expected)
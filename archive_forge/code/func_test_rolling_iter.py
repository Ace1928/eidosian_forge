from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('da', (1, 2), indirect=True)
@pytest.mark.parametrize('center', [True, False])
@pytest.mark.parametrize('size', [1, 2, 3, 7])
def test_rolling_iter(self, da: DataArray, center: bool, size: int) -> None:
    rolling_obj = da.rolling(time=size, center=center)
    rolling_obj_mean = rolling_obj.mean()
    assert len(rolling_obj.window_labels) == len(da['time'])
    assert_identical(rolling_obj.window_labels, da['time'])
    for i, (label, window_da) in enumerate(rolling_obj):
        assert label == da['time'].isel(time=i)
        actual = rolling_obj_mean.isel(time=i)
        expected = window_da.mean('time')
        np.testing.assert_allclose(actual.values, expected.values)
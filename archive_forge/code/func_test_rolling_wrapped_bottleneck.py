from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray, Dataset, set_options
from xarray.tests import (
@pytest.mark.parametrize('name', ('sum', 'mean', 'std', 'var', 'min', 'max', 'median'))
@pytest.mark.parametrize('center', (True, False, None))
@pytest.mark.parametrize('min_periods', (1, None))
@pytest.mark.parametrize('key', ('z1', 'z2'))
@pytest.mark.parametrize('backend', ['numpy'], indirect=True)
def test_rolling_wrapped_bottleneck(self, ds, name, center, min_periods, key, compute_backend) -> None:
    bn = pytest.importorskip('bottleneck', minversion='1.1')
    rolling_obj = ds.rolling(time=7, min_periods=min_periods)
    func_name = f'move_{name}'
    actual = getattr(rolling_obj, name)()
    if key == 'z1':
        expected = ds[key]
    elif key == 'z2':
        expected = getattr(bn, func_name)(ds[key].values, window=7, axis=0, min_count=min_periods)
    else:
        raise ValueError
    np.testing.assert_allclose(actual[key].values, expected)
    rolling_obj = ds.rolling(time=7, center=center)
    actual = getattr(rolling_obj, name)()['time']
    assert_allclose(actual, ds['time'])
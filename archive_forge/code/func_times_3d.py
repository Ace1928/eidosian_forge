from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import (
@pytest.fixture()
def times_3d(times):
    lons = np.linspace(0, 11, 10)
    lats = np.linspace(0, 20, 10)
    times_arr = np.random.choice(times, size=(10, 10, _NT))
    return xr.DataArray(times_arr, coords=[lons, lats, times], dims=['lon', 'lat', 'time'], name='data')
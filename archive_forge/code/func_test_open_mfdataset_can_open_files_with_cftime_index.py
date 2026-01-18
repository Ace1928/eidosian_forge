from __future__ import annotations
import pickle
from typing import TYPE_CHECKING, Any
import numpy as np
import pytest
from dask.distributed import Client, Lock
from distributed.client import futures_of
from distributed.utils_test import (  # noqa: F401
import xarray as xr
from xarray.backends.locks import HDF5_LOCK, CombinedLock, SerializableLock
from xarray.tests import (
from xarray.tests.test_backends import (
from xarray.tests.test_dataset import create_test_data
@requires_cftime
@requires_netCDF4
@pytest.mark.parametrize('parallel', (True, False))
def test_open_mfdataset_can_open_files_with_cftime_index(parallel, tmp_path):
    T = xr.cftime_range('20010101', '20010501', calendar='360_day')
    Lon = np.arange(100)
    data = np.random.random((T.size, Lon.size))
    da = xr.DataArray(data, coords={'time': T, 'Lon': Lon}, name='test')
    file_path = tmp_path / 'test.nc'
    da.to_netcdf(file_path)
    with cluster() as (s, [a, b]):
        with Client(s['address']):
            with xr.open_mfdataset(file_path, parallel=parallel) as tf:
                assert_identical(tf['test'], da)
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
def test_open_mfdataset_multiple_files_parallel_distributed(parallel, tmp_path):
    lon = np.arange(100)
    time = xr.cftime_range('20010101', periods=100, calendar='360_day')
    data = np.random.random((time.size, lon.size))
    da = xr.DataArray(data, coords={'time': time, 'lon': lon}, name='test')
    fnames = []
    for i in range(0, 100, 10):
        fname = tmp_path / f'test_{i}.nc'
        da.isel(time=slice(i, i + 10)).to_netcdf(fname)
        fnames.append(fname)
    with cluster() as (s, [a, b]):
        with Client(s['address']):
            with xr.open_mfdataset(fnames, parallel=parallel, concat_dim='time', combine='nested') as tf:
                assert_identical(tf['test'], da)
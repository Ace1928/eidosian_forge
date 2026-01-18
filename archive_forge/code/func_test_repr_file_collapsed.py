from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
@requires_netCDF4
def test_repr_file_collapsed(tmp_path) -> None:
    arr_to_store = xr.DataArray(np.arange(300, dtype=np.int64), dims='test')
    arr_to_store.to_netcdf(tmp_path / 'test.nc', engine='netcdf4')
    with xr.open_dataarray(tmp_path / 'test.nc') as arr, xr.set_options(display_expand_data=False):
        actual = repr(arr)
        expected = dedent('        <xarray.DataArray (test: 300)> Size: 2kB\n        [300 values with dtype=int64]\n        Dimensions without coordinates: test')
        assert actual == expected
        arr_loaded = arr.compute()
        actual = arr_loaded.__repr__()
        expected = dedent('        <xarray.DataArray (test: 300)> Size: 2kB\n        0 1 2 3 4 5 6 7 8 9 10 11 12 ... 288 289 290 291 292 293 294 295 296 297 298 299\n        Dimensions without coordinates: test')
        assert actual == expected
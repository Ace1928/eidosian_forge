from __future__ import annotations
import contextlib
import gzip
import itertools
import math
import os.path
import pickle
import platform
import re
import shutil
import sys
import tempfile
import uuid
import warnings
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack
from io import BytesIO
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.errors import OutOfBoundsDatetime
import xarray as xr
from xarray import (
from xarray.backends.common import robust_getitem
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.backends.netCDF4_ import (
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.scipy_ import ScipyBackendEntrypoint
from xarray.coding.cftime_offsets import cftime_range
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_coding_times import (
from xarray.tests.test_dataset import (
@requires_zarr
class TestZarrRegionAuto:

    def test_zarr_region_auto_all(self, tmp_path):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds.to_zarr(tmp_path / 'test.zarr')
        ds_region = 1 + ds.isel(x=slice(2, 4), y=slice(6, 8))
        ds_region.to_zarr(tmp_path / 'test.zarr', region='auto')
        ds_updated = xr.open_zarr(tmp_path / 'test.zarr')
        expected = ds.copy()
        expected['test'][2:4, 6:8] += 1
        assert_identical(ds_updated, expected)

    def test_zarr_region_auto_mixed(self, tmp_path):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds.to_zarr(tmp_path / 'test.zarr')
        ds_region = 1 + ds.isel(x=slice(2, 4), y=slice(6, 8))
        ds_region.to_zarr(tmp_path / 'test.zarr', region={'x': 'auto', 'y': slice(6, 8)})
        ds_updated = xr.open_zarr(tmp_path / 'test.zarr')
        expected = ds.copy()
        expected['test'][2:4, 6:8] += 1
        assert_identical(ds_updated, expected)

    def test_zarr_region_auto_noncontiguous(self, tmp_path):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds.to_zarr(tmp_path / 'test.zarr')
        ds_region = 1 + ds.isel(x=[0, 2, 3], y=[5, 6])
        with pytest.raises(ValueError):
            ds_region.to_zarr(tmp_path / 'test.zarr', region={'x': 'auto', 'y': 'auto'})

    def test_zarr_region_auto_new_coord_vals(self, tmp_path):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds.to_zarr(tmp_path / 'test.zarr')
        x = np.arange(5, 55, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds_region = 1 + ds.isel(x=slice(2, 4), y=slice(6, 8))
        with pytest.raises(KeyError):
            ds_region.to_zarr(tmp_path / 'test.zarr', region={'x': 'auto', 'y': 'auto'})

    def test_zarr_region_index_write(self, tmp_path):
        from xarray.backends.zarr import ZarrStore
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        region_slice = dict(x=slice(2, 4), y=slice(6, 8))
        ds_region = 1 + ds.isel(region_slice)
        ds.to_zarr(tmp_path / 'test.zarr')
        region: Mapping[str, slice] | Literal['auto']
        for region in [region_slice, 'auto']:
            with patch.object(ZarrStore, 'set_variables', side_effect=ZarrStore.set_variables, autospec=True) as mock:
                ds_region.to_zarr(tmp_path / 'test.zarr', region=region, mode='r+')
                for call in mock.call_args_list:
                    written_variables = call.args[1].keys()
                    assert 'test' in written_variables
                    assert 'x' not in written_variables
                    assert 'y' not in written_variables

    def test_zarr_region_append(self, tmp_path):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset({'test': xr.DataArray(data, dims=('x', 'y'), coords={'x': x, 'y': y})})
        ds.to_zarr(tmp_path / 'test.zarr')
        x_new = np.arange(40, 70, 10)
        data_new = np.ones((3, 10))
        ds_new = xr.Dataset({'test': xr.DataArray(data_new, dims=('x', 'y'), coords={'x': x_new, 'y': y})})
        with pytest.raises(ValueError):
            ds_new.to_zarr(tmp_path / 'test.zarr', mode='a', append_dim='x', region='auto')
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
@requires_scipy_or_netCDF4
@requires_dask
class TestOpenMFDatasetWithDataVarsAndCoordsKw:
    coord_name = 'lon'
    var_name = 'v1'

    @contextlib.contextmanager
    def setup_files_and_datasets(self, fuzz=0):
        ds1, ds2 = self.gen_datasets_with_common_coord_and_time()
        ds1['x'] = ds1.x + fuzz
        with create_tmp_file() as tmpfile1:
            with create_tmp_file() as tmpfile2:
                ds1.to_netcdf(tmpfile1)
                ds2.to_netcdf(tmpfile2)
                yield ([tmpfile1, tmpfile2], [ds1, ds2])

    def gen_datasets_with_common_coord_and_time(self):
        nx = 10
        nt = 10
        x = np.arange(nx)
        t1 = np.arange(nt)
        t2 = np.arange(nt, 2 * nt, 1)
        v1 = np.random.randn(nt, nx)
        v2 = np.random.randn(nt, nx)
        ds1 = Dataset(data_vars={self.var_name: (['t', 'x'], v1), self.coord_name: ('x', 2 * x)}, coords={'t': (['t'], t1), 'x': (['x'], x)})
        ds2 = Dataset(data_vars={self.var_name: (['t', 'x'], v2), self.coord_name: ('x', 2 * x)}, coords={'t': (['t'], t2), 'x': (['x'], x)})
        return (ds1, ds2)

    @pytest.mark.parametrize('combine, concat_dim', [('nested', 't'), ('by_coords', None)])
    @pytest.mark.parametrize('opt', ['all', 'minimal', 'different'])
    @pytest.mark.parametrize('join', ['outer', 'inner', 'left', 'right'])
    def test_open_mfdataset_does_same_as_concat(self, combine, concat_dim, opt, join) -> None:
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            if combine == 'by_coords':
                files.reverse()
            with open_mfdataset(files, data_vars=opt, combine=combine, concat_dim=concat_dim, join=join) as ds:
                ds_expect = xr.concat([ds1, ds2], data_vars=opt, dim='t', join=join)
                assert_identical(ds, ds_expect)

    @pytest.mark.parametrize(['combine_attrs', 'attrs', 'expected', 'expect_error'], (pytest.param('drop', [{'a': 1}, {'a': 2}], {}, False, id='drop'), pytest.param('override', [{'a': 1}, {'a': 2}], {'a': 1}, False, id='override'), pytest.param('no_conflicts', [{'a': 1}, {'a': 2}], None, True, id='no_conflicts'), pytest.param('identical', [{'a': 1, 'b': 2}, {'a': 1, 'c': 3}], None, True, id='identical'), pytest.param('drop_conflicts', [{'a': 1, 'b': 2}, {'b': -1, 'c': 3}], {'a': 1, 'c': 3}, False, id='drop_conflicts')))
    def test_open_mfdataset_dataset_combine_attrs(self, combine_attrs, attrs, expected, expect_error):
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds.attrs = attrs[i]
                ds.close()
                ds.to_netcdf(f)
            if expect_error:
                with pytest.raises(xr.MergeError):
                    xr.open_mfdataset(files, combine='nested', concat_dim='t', combine_attrs=combine_attrs)
            else:
                with xr.open_mfdataset(files, combine='nested', concat_dim='t', combine_attrs=combine_attrs) as ds:
                    assert ds.attrs == expected

    def test_open_mfdataset_dataset_attr_by_coords(self) -> None:
        """
        Case when an attribute differs across the multiple files
        """
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds.attrs['test_dataset_attr'] = 10 + i
                ds.close()
                ds.to_netcdf(f)
            with xr.open_mfdataset(files, combine='nested', concat_dim='t') as ds:
                assert ds.test_dataset_attr == 10

    def test_open_mfdataset_dataarray_attr_by_coords(self) -> None:
        """
        Case when an attribute of a member DataArray differs across the multiple files
        """
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds['v1'].attrs['test_dataarray_attr'] = i
                ds.close()
                ds.to_netcdf(f)
            with xr.open_mfdataset(files, combine='nested', concat_dim='t') as ds:
                assert ds['v1'].test_dataarray_attr == 0

    @pytest.mark.parametrize('combine, concat_dim', [('nested', 't'), ('by_coords', None)])
    @pytest.mark.parametrize('opt', ['all', 'minimal', 'different'])
    def test_open_mfdataset_exact_join_raises_error(self, combine, concat_dim, opt) -> None:
        with self.setup_files_and_datasets(fuzz=0.1) as (files, [ds1, ds2]):
            if combine == 'by_coords':
                files.reverse()
            with pytest.raises(ValueError, match='cannot align objects.*join.*exact.*'):
                open_mfdataset(files, data_vars=opt, combine=combine, concat_dim=concat_dim, join='exact')

    def test_common_coord_when_datavars_all(self) -> None:
        opt: Final = 'all'
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            with open_mfdataset(files, data_vars=opt, combine='nested', concat_dim='t') as ds:
                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape
                var_shape = ds[self.var_name].shape
                assert var_shape == coord_shape
                assert coord_shape1 != coord_shape
                assert coord_shape2 != coord_shape

    def test_common_coord_when_datavars_minimal(self) -> None:
        opt: Final = 'minimal'
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            with open_mfdataset(files, data_vars=opt, combine='nested', concat_dim='t') as ds:
                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape
                var_shape = ds[self.var_name].shape
                assert var_shape != coord_shape
                assert coord_shape1 == coord_shape
                assert coord_shape2 == coord_shape

    def test_invalid_data_vars_value_should_fail(self) -> None:
        with self.setup_files_and_datasets() as (files, _):
            with pytest.raises(ValueError):
                with open_mfdataset(files, data_vars='minimum', combine='by_coords'):
                    pass
            with pytest.raises(ValueError):
                with open_mfdataset(files, coords='minimum', combine='by_coords'):
                    pass
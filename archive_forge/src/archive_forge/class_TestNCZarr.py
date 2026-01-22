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
@requires_netCDF4
class TestNCZarr:

    @property
    def netcdfc_version(self):
        return Version(nc4.getlibversion().split()[0].split('-development')[0])

    def _create_nczarr(self, filename):
        if self.netcdfc_version < Version('4.8.1'):
            pytest.skip('requires netcdf-c>=4.8.1')
        if platform.system() == 'Windows' and self.netcdfc_version == Version('4.8.1'):
            pytest.skip('netcdf-c==4.8.1 has issues on Windows')
        ds = create_test_data()
        ds = ds.drop_vars('dim3')
        ds.to_netcdf(f'file://{filename}#mode=nczarr')
        return ds

    def test_open_nczarr(self) -> None:
        with create_tmp_file(suffix='.zarr') as tmp:
            expected = self._create_nczarr(tmp)
            actual = xr.open_zarr(tmp, consolidated=False)
            assert_identical(expected, actual)

    def test_overwriting_nczarr(self) -> None:
        with create_tmp_file(suffix='.zarr') as tmp:
            ds = self._create_nczarr(tmp)
            expected = ds[['var1']]
            expected.to_zarr(tmp, mode='w')
            actual = xr.open_zarr(tmp, consolidated=False)
            assert_identical(expected, actual)

    @pytest.mark.parametrize('mode', ['a', 'r+'])
    @pytest.mark.filterwarnings('ignore:.*non-consolidated metadata.*')
    def test_raise_writing_to_nczarr(self, mode) -> None:
        if self.netcdfc_version > Version('4.8.1'):
            pytest.skip('netcdf-c>4.8.1 adds the _ARRAY_DIMENSIONS attribute')
        with create_tmp_file(suffix='.zarr') as tmp:
            ds = self._create_nczarr(tmp)
            with pytest.raises(KeyError, match='missing the attribute `_ARRAY_DIMENSIONS`,'):
                ds.to_zarr(tmp, mode=mode)
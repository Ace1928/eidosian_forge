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
@pytest.mark.parametrize('consolidated', [True, False, None])
@pytest.mark.parametrize('write_empty', [True, False, None])
def test_write_empty(self, consolidated: bool | None, write_empty: bool | None) -> None:
    if write_empty is False:
        expected = ['0.1.0', '1.1.0']
    else:
        expected = ['0.0.0', '0.0.1', '0.1.0', '0.1.1', '1.0.0', '1.0.1', '1.1.0', '1.1.1']
    ds = xr.Dataset(data_vars={'test': (('Z', 'Y', 'X'), np.array([np.nan, np.nan, 1.0, np.nan]).reshape((1, 2, 2)))})
    if has_dask:
        ds['test'] = ds['test'].chunk(1)
        encoding = None
    else:
        encoding = {'test': {'chunks': (1, 1, 1)}}
    with self.temp_dir() as (d, store):
        ds.to_zarr(store, mode='w', encoding=encoding, write_empty_chunks=write_empty)
        with self.roundtrip_dir(ds, store, {'mode': 'a', 'append_dim': 'Z', 'write_empty_chunks': write_empty}) as a_ds:
            expected_ds = xr.concat([ds, ds], dim='Z')
            assert_identical(a_ds, expected_ds)
            ls = listdir(os.path.join(store, 'test'))
            assert set(expected) == set([file for file in ls if file[0] != '.'])
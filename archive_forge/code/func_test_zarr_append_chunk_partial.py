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
@requires_dask
def test_zarr_append_chunk_partial(tmp_path):
    t_coords = np.array([np.datetime64('2020-01-01').astype('datetime64[ns]')])
    data = np.ones((10, 10))
    da = xr.DataArray(data.reshape((-1, 10, 10)), dims=['time', 'x', 'y'], coords={'time': t_coords}, name='foo')
    da.to_zarr(tmp_path / 'foo.zarr', mode='w', encoding={'foo': {'chunks': (5, 5, 1)}})
    new_time = np.array([np.datetime64('2021-01-01').astype('datetime64[ns]')])
    da2 = xr.DataArray(data.reshape((-1, 10, 10)), dims=['time', 'x', 'y'], coords={'time': new_time}, name='foo')
    with pytest.raises(ValueError, match='encoding was provided'):
        da2.to_zarr(tmp_path / 'foo.zarr', append_dim='time', mode='a', encoding={'foo': {'chunks': (1, 1, 1)}})
    with pytest.raises(ValueError, match='Specified zarr chunks'):
        da2.chunk({'x': 1, 'y': 1, 'time': 1}).to_zarr(tmp_path / 'foo.zarr', append_dim='time', mode='a')
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
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
@pytest.mark.parametrize('calendar', _STANDARD_CALENDARS)
def test_use_cftime_standard_calendar_default_in_range(calendar) -> None:
    x = [0, 1]
    time = [0, 720]
    units_date = '2000-01-01'
    units = 'days since 2000-01-01'
    original = DataArray(x, [('time', time)], name='x').to_dataset()
    for v in ['x', 'time']:
        original[v].attrs['units'] = units
        original[v].attrs['calendar'] = calendar
    x_timedeltas = np.array(x).astype('timedelta64[D]')
    time_timedeltas = np.array(time).astype('timedelta64[D]')
    decoded_x = np.datetime64(units_date, 'ns') + x_timedeltas
    decoded_time = np.datetime64(units_date, 'ns') + time_timedeltas
    expected_x = DataArray(decoded_x, [('time', decoded_time)], name='x')
    expected_time = DataArray(decoded_time, [('time', decoded_time)], name='time')
    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with warnings.catch_warnings(record=True) as record:
            with open_dataset(tmp_file) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            _assert_no_dates_out_of_range_warning(record)
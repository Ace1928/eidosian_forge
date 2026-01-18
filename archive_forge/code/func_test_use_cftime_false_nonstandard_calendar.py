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
@pytest.mark.parametrize('calendar', _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize('units_year', [1500, 2000, 2500])
def test_use_cftime_false_nonstandard_calendar(calendar, units_year) -> None:
    x = [0, 1]
    time = [0, 720]
    units = f'days since {units_year}'
    original = DataArray(x, [('time', time)], name='x').to_dataset()
    for v in ['x', 'time']:
        original[v].attrs['units'] = units
        original[v].attrs['calendar'] = calendar
    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with pytest.raises((OutOfBoundsDatetime, ValueError)):
            open_dataset(tmp_file, use_cftime=False)
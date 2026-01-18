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
@requires_scipy
def test_open_fileobj(self) -> None:
    expected = create_test_data().drop_vars('dim3')
    expected.attrs['foo'] = 'bar'
    with create_tmp_file() as tmp_file:
        expected.to_netcdf(tmp_file, engine='h5netcdf')
        with open(tmp_file, 'rb') as f:
            with open_dataset(f, engine='h5netcdf') as actual:
                assert_identical(expected, actual)
            f.seek(0)
            with open_dataset(f) as actual:
                assert_identical(expected, actual)
            f.seek(0)
            with BytesIO(f.read()) as bio:
                with open_dataset(bio, engine='h5netcdf') as actual:
                    assert_identical(expected, actual)
            f.seek(0)
            with pytest.raises(TypeError, match='not a valid NetCDF 3'):
                open_dataset(f, engine='scipy')
        with open(tmp_file, 'rb') as f:
            f.seek(8)
            open_dataset(f)
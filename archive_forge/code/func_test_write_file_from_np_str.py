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
@requires_netCDF4
@pytest.mark.parametrize('str_type', (str, np.str_))
def test_write_file_from_np_str(str_type, tmpdir) -> None:
    scenarios = [str_type(v) for v in ['scenario_a', 'scenario_b', 'scenario_c']]
    years = range(2015, 2100 + 1)
    tdf = pd.DataFrame(data=np.random.random((len(scenarios), len(years))), columns=years, index=scenarios)
    tdf.index.name = 'scenario'
    tdf.columns.name = 'year'
    tdf = tdf.stack()
    tdf.name = 'tas'
    txr = tdf.to_xarray()
    txr.to_netcdf(tmpdir.join('test.nc'))
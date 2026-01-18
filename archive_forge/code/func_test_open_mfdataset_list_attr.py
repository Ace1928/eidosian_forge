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
@requires_dask
def test_open_mfdataset_list_attr() -> None:
    """
    Case when an attribute of type list differs across the multiple files
    """
    from netCDF4 import Dataset
    with create_tmp_files(2) as nfiles:
        for i in range(2):
            with Dataset(nfiles[i], 'w') as f:
                f.createDimension('x', 3)
                vlvar = f.createVariable('test_var', np.int32, 'x')
                vlvar.test_attr = [f'string a {i}', f'string b {i}']
                vlvar[:] = np.arange(3)
        with open_dataset(nfiles[0]) as ds1:
            with open_dataset(nfiles[1]) as ds2:
                original = xr.concat([ds1, ds2], dim='x')
                with xr.open_mfdataset([nfiles[0], nfiles[1]], combine='nested', concat_dim='x') as actual:
                    assert_identical(actual, original)
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
def test_open_subgroup(self) -> None:
    with create_tmp_file() as tmp_file:
        rootgrp = nc4.Dataset(tmp_file, 'w')
        foogrp = rootgrp.createGroup('foo')
        bargrp = foogrp.createGroup('bar')
        ds = bargrp
        ds.createDimension('time', size=10)
        x = np.arange(10)
        ds.createVariable('x', np.int32, dimensions=('time',))
        ds.variables['x'][:] = x
        rootgrp.close()
        expected = Dataset()
        expected['x'] = ('time', x)
        for group in ('foo/bar', '/foo/bar', 'foo/bar/', '/foo/bar/'):
            with self.open(tmp_file, group=group) as actual:
                assert_equal(actual['x'], expected['x'])
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
def validate_array_type(self, ds):

    def find_and_validate_array(obj):
        if hasattr(obj, 'array'):
            if isinstance(obj.array, indexing.ExplicitlyIndexed):
                find_and_validate_array(obj.array)
            elif isinstance(obj.array, np.ndarray):
                assert isinstance(obj, indexing.NumpyIndexingAdapter)
            elif isinstance(obj.array, dask_array_type):
                assert isinstance(obj, indexing.DaskIndexingAdapter)
            elif isinstance(obj.array, pd.Index):
                assert isinstance(obj, indexing.PandasIndexingAdapter)
            else:
                raise TypeError(f'{type(obj.array)} is wrapped by {type(obj)}')
    for k, v in ds.variables.items():
        find_and_validate_array(v._data)
from __future__ import annotations
import functools
import operator
import os
from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import (
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
def _nc4_dtype(var):
    if 'dtype' in var.encoding:
        dtype = var.encoding.pop('dtype')
        _check_encoding_dtype_is_vlen_string(dtype)
    elif coding.strings.is_unicode_dtype(var.dtype):
        dtype = str
    elif var.dtype.kind in ['i', 'u', 'f', 'c', 'S']:
        dtype = var.dtype
    else:
        raise ValueError(f'unsupported dtype for netCDF4 variable: {var.dtype}')
    return dtype
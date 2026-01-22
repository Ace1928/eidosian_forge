from __future__ import annotations
import functools
import io
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import HDF5_LOCK, combine_locks, ensure_lock, get_write_lock
from xarray.backends.netCDF4_ import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
def _read_attributes(h5netcdf_var):
    attrs = {}
    for k, v in h5netcdf_var.attrs.items():
        if k not in ['_FillValue', 'missing_value']:
            if isinstance(v, bytes):
                try:
                    v = v.decode('utf-8')
                except UnicodeDecodeError:
                    emit_user_level_warning(f"'utf-8' codec can't decode bytes for attribute {k!r} of h5netcdf object {h5netcdf_var.name!r}, returning bytes undecoded.", UnicodeWarning)
        attrs[k] = v
    return attrs
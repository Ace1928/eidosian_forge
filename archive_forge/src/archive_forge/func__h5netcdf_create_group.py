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
def _h5netcdf_create_group(dataset, name):
    return dataset.create_group(name)
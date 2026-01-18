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

    Backend for netCDF files based on the h5netcdf package.

    It can open ".nc", ".nc4", ".cdf" files but will only be
    selected as the default if the "netcdf4" engine is not available.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info.
    It will not be detected as valid backend for such files, so make
    sure to specify ``engine="h5netcdf"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://h5netcdf.org

    See Also
    --------
    backends.H5NetCDFStore
    backends.NetCDF4BackendEntrypoint
    backends.ScipyBackendEntrypoint
    
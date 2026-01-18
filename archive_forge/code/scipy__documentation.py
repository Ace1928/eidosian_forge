from __future__ import annotations
import gzip
import io
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import ensure_lock, get_write_lock
from xarray.backends.netcdf3 import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable

    Backend for netCDF files based on the scipy package.

    It can open ".nc", ".nc4", ".cdf" and ".gz" files but will only be
    selected as the default if the "netcdf4" and "h5netcdf" engines are
    not available. It has the advantage that is is a lightweight engine
    that has no system requirements (unlike netcdf4 and h5netcdf).

    Additionally it can open gizp compressed (".gz") files.

    For more information about the underlying library, visit:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html

    See Also
    --------
    backends.ScipyDataStore
    backends.NetCDF4BackendEntrypoint
    backends.H5netcdfBackendEntrypoint
    
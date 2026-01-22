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
def _open_scipy_netcdf(filename, mode, mmap, version):
    import scipy.io
    if isinstance(filename, str) and filename.endswith('.gz'):
        try:
            return scipy.io.netcdf_file(gzip.open(filename), mode=mode, mmap=mmap, version=version)
        except TypeError as e:
            errmsg = e.args[0]
            if 'is not a valid NetCDF 3 file' in errmsg:
                raise ValueError('gzipped file loading only supports NetCDF 3 files.')
            else:
                raise
    if isinstance(filename, bytes) and filename.startswith(b'CDF'):
        filename = io.BytesIO(filename)
    try:
        return scipy.io.netcdf_file(filename, mode=mode, mmap=mmap, version=version)
    except TypeError as e:
        errmsg = e.args[0]
        if 'is not a valid NetCDF 3 file' in errmsg:
            msg = '\n            If this is a NetCDF4 file, you may need to install the\n            netcdf4 library, e.g.,\n\n            $ pip install netcdf4\n            '
            errmsg += msg
            raise TypeError(errmsg)
        else:
            raise
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
class ScipyBackendEntrypoint(BackendEntrypoint):
    """
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
    """
    description = 'Open netCDF files (.nc, .nc4, .cdf and .gz) using scipy in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.ScipyBackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        magic_number = try_read_magic_number_from_file_or_path(filename_or_obj)
        if magic_number is not None and magic_number.startswith(b'\x1f\x8b'):
            with gzip.open(filename_or_obj) as f:
                magic_number = try_read_magic_number_from_file_or_path(f)
        if magic_number is not None:
            return magic_number.startswith(b'CDF')
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {'.nc', '.nc4', '.cdf', '.gz'}
        return False

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, mode='r', format=None, group=None, mmap=None, lock=None) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = ScipyDataStore(filename_or_obj, mode=mode, format=format, group=group, mmap=mmap, lock=lock)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds
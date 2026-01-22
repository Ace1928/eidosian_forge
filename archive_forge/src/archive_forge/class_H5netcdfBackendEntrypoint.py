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
class H5netcdfBackendEntrypoint(BackendEntrypoint):
    """
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
    """
    description = 'Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using h5netcdf in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.H5netcdfBackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        magic_number = try_read_magic_number_from_file_or_path(filename_or_obj)
        if magic_number is not None:
            return magic_number.startswith(b'\x89HDF\r\n\x1a\n')
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {'.nc', '.nc4', '.cdf'}
        return False

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, format=None, group=None, lock=None, invalid_netcdf=None, phony_dims=None, decode_vlen_strings=True, driver=None, driver_kwds=None) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = H5NetCDFStore.open(filename_or_obj, format=format, group=group, lock=lock, invalid_netcdf=invalid_netcdf, phony_dims=phony_dims, decode_vlen_strings=decode_vlen_strings, driver=driver, driver_kwds=driver_kwds)
        store_entrypoint = StoreBackendEntrypoint()
        ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds

    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs) -> DataTree:
        from h5netcdf.legacyapi import Dataset as ncDataset
        return _open_datatree_netcdf(ncDataset, filename_or_obj, **kwargs)
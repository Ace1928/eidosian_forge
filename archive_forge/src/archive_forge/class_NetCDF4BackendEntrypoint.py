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
class NetCDF4BackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the netCDF4 package.

    It can open ".nc", ".nc4", ".cdf" files and will be chosen
    as default for these files.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info.
    It will not be detected as valid backend for such files, so make
    sure to specify ``engine="netcdf4"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://unidata.github.io/netcdf4-python

    See Also
    --------
    backends.NetCDF4DataStore
    backends.H5netcdfBackendEntrypoint
    backends.ScipyBackendEntrypoint
    """
    description = 'Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using netCDF4 in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.NetCDF4BackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        if isinstance(filename_or_obj, str) and is_remote_uri(filename_or_obj):
            return True
        magic_number = try_read_magic_number_from_path(filename_or_obj)
        if magic_number is not None:
            return magic_number.startswith((b'CDF', b'\x89HDF\r\n\x1a\n'))
        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {'.nc', '.nc4', '.cdf'}
        return False

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, group=None, mode='r', format='NETCDF4', clobber=True, diskless=False, persist=False, lock=None, autoclose=False) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = NetCDF4DataStore.open(filename_or_obj, mode=mode, format=format, group=group, clobber=clobber, diskless=diskless, persist=persist, lock=lock, autoclose=autoclose)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds

    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs) -> DataTree:
        from netCDF4 import Dataset as ncDataset
        return _open_datatree_netcdf(ncDataset, filename_or_obj, **kwargs)
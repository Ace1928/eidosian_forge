from __future__ import annotations
import os
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import partial
from io import BytesIO
from numbers import Number
from typing import (
import numpy as np
from xarray import backends, conventions
from xarray.backends import plugins
from xarray.backends.common import (
from xarray.backends.locks import _get_scheduler
from xarray.backends.zarr import open_zarr
from xarray.core import indexing
from xarray.core.combine import (
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, _get_chunk, _maybe_chunk
from xarray.core.indexes import Index
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import is_remote_uri
from xarray.namedarray.daskmanager import DaskManager
from xarray.namedarray.parallelcompat import guess_chunkmanager
def to_netcdf(dataset: Dataset, path_or_file: str | os.PathLike | None=None, mode: Literal['w', 'a']='w', format: T_NetcdfTypes | None=None, group: str | None=None, engine: T_NetcdfEngine | None=None, encoding: Mapping[Hashable, Mapping[str, Any]] | None=None, unlimited_dims: Iterable[Hashable] | None=None, compute: bool=True, multifile: bool=False, invalid_netcdf: bool=False) -> tuple[ArrayWriter, AbstractDataStore] | bytes | Delayed | None:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    if isinstance(path_or_file, os.PathLike):
        path_or_file = os.fspath(path_or_file)
    if encoding is None:
        encoding = {}
    if path_or_file is None:
        if engine is None:
            engine = 'scipy'
        elif engine != 'scipy':
            raise ValueError(f"invalid engine for creating bytes with to_netcdf: {engine!r}. Only the default engine or engine='scipy' is supported")
        if not compute:
            raise NotImplementedError('to_netcdf() with compute=False is not yet implemented when returning bytes')
    elif isinstance(path_or_file, str):
        if engine is None:
            engine = _get_default_engine(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    else:
        engine = 'scipy'
    _validate_dataset_names(dataset)
    _validate_attrs(dataset, invalid_netcdf=invalid_netcdf and engine == 'h5netcdf')
    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError(f'unrecognized engine for to_netcdf: {engine!r}')
    if format is not None:
        format = format.upper()
    scheduler = _get_scheduler()
    have_chunks = any((v.chunks is not None for v in dataset.variables.values()))
    autoclose = have_chunks and scheduler in ['distributed', 'multiprocessing']
    if autoclose and engine == 'scipy':
        raise NotImplementedError(f"Writing netCDF files with the {engine} backend is not currently supported with dask's {scheduler} scheduler")
    target = path_or_file if path_or_file is not None else BytesIO()
    kwargs = dict(autoclose=True) if autoclose else {}
    if invalid_netcdf:
        if engine == 'h5netcdf':
            kwargs['invalid_netcdf'] = invalid_netcdf
        else:
            raise ValueError(f"unrecognized option 'invalid_netcdf' for engine {engine}")
    store = store_open(target, mode, format, group, **kwargs)
    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get('unlimited_dims', None)
    if unlimited_dims is not None:
        if isinstance(unlimited_dims, str) or not isinstance(unlimited_dims, Iterable):
            unlimited_dims = [unlimited_dims]
        else:
            unlimited_dims = list(unlimited_dims)
    writer = ArrayWriter()
    try:
        dump_to_store(dataset, store, writer, encoding=encoding, unlimited_dims=unlimited_dims)
        if autoclose:
            store.close()
        if multifile:
            return (writer, store)
        writes = writer.sync(compute=compute)
        if isinstance(target, BytesIO):
            store.sync()
            return target.getvalue()
    finally:
        if not multifile and compute:
            store.close()
    if not compute:
        import dask
        return dask.delayed(_finalize_store)(writes, store)
    return None
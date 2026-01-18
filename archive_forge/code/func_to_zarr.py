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
def to_zarr(dataset: Dataset, store: MutableMapping | str | os.PathLike[str] | None=None, chunk_store: MutableMapping | str | os.PathLike | None=None, mode: ZarrWriteModes | None=None, synchronizer=None, group: str | None=None, encoding: Mapping | None=None, *, compute: bool=True, consolidated: bool | None=None, append_dim: Hashable | None=None, region: Mapping[str, slice | Literal['auto']] | Literal['auto'] | None=None, safe_chunks: bool=True, storage_options: dict[str, str] | None=None, zarr_version: int | None=None, write_empty_chunks: bool | None=None, chunkmanager_store_kwargs: dict[str, Any] | None=None) -> backends.ZarrStore | Delayed:
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """
    for v in dataset.variables.values():
        if v.size == 0:
            v.load()
    store = _normalize_path(store)
    chunk_store = _normalize_path(chunk_store)
    if storage_options is None:
        mapper = store
        chunk_mapper = chunk_store
    else:
        from fsspec import get_mapper
        if not isinstance(store, str):
            raise ValueError(f'store must be a string to use storage_options. Got {type(store)}')
        mapper = get_mapper(store, **storage_options)
        if chunk_store is not None:
            chunk_mapper = get_mapper(chunk_store, **storage_options)
        else:
            chunk_mapper = chunk_store
    if encoding is None:
        encoding = {}
    if mode is None:
        if append_dim is not None:
            mode = 'a'
        elif region is not None:
            mode = 'r+'
        else:
            mode = 'w-'
    if mode not in ['a', 'a-'] and append_dim is not None:
        raise ValueError("cannot set append_dim unless mode='a' or mode=None")
    if mode not in ['a', 'a-', 'r+'] and region is not None:
        raise ValueError("cannot set region unless mode='a', mode='a-', mode='r+' or mode=None")
    if mode not in ['w', 'w-', 'a', 'a-', 'r+']:
        raise ValueError(f"The only supported options for mode are 'w', 'w-', 'a', 'a-', and 'r+', but mode={mode!r}")
    _validate_dataset_names(dataset)
    if region is not None:
        open_kwargs = dict(store=store, synchronizer=synchronizer, group=group, consolidated=consolidated, storage_options=storage_options, zarr_version=zarr_version)
        region = _validate_and_autodetect_region(dataset, region, mode, open_kwargs)
        dataset = dataset.drop_vars(dataset.indexes)
        if append_dim is not None and append_dim in region:
            raise ValueError(f'cannot list the same dimension in both ``append_dim`` and ``region`` with to_zarr(), got {append_dim} in both')
    if zarr_version is None:
        zarr_version = int(getattr(store, '_store_version', 2))
    if consolidated is None and zarr_version > 2:
        consolidated = False
    if mode == 'r+':
        already_consolidated = consolidated
        consolidate_on_close = False
    else:
        already_consolidated = False
        consolidate_on_close = consolidated or consolidated is None
    zstore = backends.ZarrStore.open_group(store=mapper, mode=mode, synchronizer=synchronizer, group=group, consolidated=already_consolidated, consolidate_on_close=consolidate_on_close, chunk_store=chunk_mapper, append_dim=append_dim, write_region=region, safe_chunks=safe_chunks, stacklevel=4, zarr_version=zarr_version, write_empty=write_empty_chunks)
    if mode in ['a', 'a-', 'r+']:
        _validate_datatypes_for_zarr_append(zstore, dataset)
        if append_dim is not None:
            existing_dims = zstore.get_dimensions()
            if append_dim not in existing_dims:
                raise ValueError(f'append_dim={append_dim!r} does not match any existing dataset dimensions {existing_dims}')
        existing_var_names = set(zstore.zarr_group.array_keys())
        for var_name in existing_var_names:
            if var_name in encoding.keys():
                raise ValueError(f'variable {var_name!r} already exists, but encoding was provided')
        if mode == 'r+':
            new_names = [k for k in dataset.variables if k not in existing_var_names]
            if new_names:
                raise ValueError(f"dataset contains non-pre-existing variables {new_names}, which is not allowed in ``xarray.Dataset.to_zarr()`` with mode='r+'. To allow writing new variables, set mode='a'.")
    writer = ArrayWriter()
    dump_to_store(dataset, zstore, writer, encoding=encoding)
    writes = writer.sync(compute=compute, chunkmanager_store_kwargs=chunkmanager_store_kwargs)
    if compute:
        _finalize_store(writes, zstore)
    else:
        import dask
        return dask.delayed(_finalize_store)(writes, zstore)
    return zstore
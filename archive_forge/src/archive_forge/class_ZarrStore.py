from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr"""
    __slots__ = ('zarr_group', '_append_dim', '_consolidate_on_close', '_group', '_mode', '_read_only', '_synchronizer', '_write_region', '_safe_chunks', '_write_empty', '_close_store_on_close')

    @classmethod
    def open_group(cls, store, mode: ZarrWriteModes='r', synchronizer=None, group=None, consolidated=False, consolidate_on_close=False, chunk_store=None, storage_options=None, append_dim=None, write_region=None, safe_chunks=True, stacklevel=2, zarr_version=None, write_empty: bool | None=None):
        import zarr
        if isinstance(store, os.PathLike):
            store = os.fspath(store)
        if zarr_version is None:
            zarr_version = getattr(store, '_store_version', 2)
        open_kwargs = dict(mode='a' if mode == 'a-' else mode, synchronizer=synchronizer, path=group)
        open_kwargs['storage_options'] = storage_options
        if zarr_version > 2:
            open_kwargs['zarr_version'] = zarr_version
            if consolidated or consolidate_on_close:
                raise ValueError(f'consolidated metadata has not been implemented for zarr version {zarr_version} yet. Set consolidated=False for zarr version {zarr_version}. See also https://github.com/zarr-developers/zarr-specs/issues/136')
            if consolidated is None:
                consolidated = False
        if chunk_store is not None:
            open_kwargs['chunk_store'] = chunk_store
            if consolidated is None:
                consolidated = False
        if consolidated is None:
            try:
                zarr_group = zarr.open_consolidated(store, **open_kwargs)
            except KeyError:
                try:
                    zarr_group = zarr.open_group(store, **open_kwargs)
                    warnings.warn('Failed to open Zarr store with consolidated metadata, but successfully read with non-consolidated metadata. This is typically much slower for opening a dataset. To silence this warning, consider:\n1. Consolidating metadata in this existing store with zarr.consolidate_metadata().\n2. Explicitly setting consolidated=False, to avoid trying to read consolidate metadata, or\n3. Explicitly setting consolidated=True, to raise an error in this case instead of falling back to try reading non-consolidated metadata.', RuntimeWarning, stacklevel=stacklevel)
                except zarr.errors.GroupNotFoundError:
                    raise FileNotFoundError(f"No such file or directory: '{store}'")
        elif consolidated:
            zarr_group = zarr.open_consolidated(store, **open_kwargs)
        else:
            zarr_group = zarr.open_group(store, **open_kwargs)
        close_store_on_close = zarr_group.store is not store
        return cls(zarr_group, mode, consolidate_on_close, append_dim, write_region, safe_chunks, write_empty, close_store_on_close)

    def __init__(self, zarr_group, mode=None, consolidate_on_close=False, append_dim=None, write_region=None, safe_chunks=True, write_empty: bool | None=None, close_store_on_close: bool=False):
        self.zarr_group = zarr_group
        self._read_only = self.zarr_group.read_only
        self._synchronizer = self.zarr_group.synchronizer
        self._group = self.zarr_group.path
        self._mode = mode
        self._consolidate_on_close = consolidate_on_close
        self._append_dim = append_dim
        self._write_region = write_region
        self._safe_chunks = safe_chunks
        self._write_empty = write_empty
        self._close_store_on_close = close_store_on_close

    @property
    def ds(self):
        return self.zarr_group

    def open_store_variable(self, name, zarr_array):
        data = indexing.LazilyIndexedArray(ZarrArrayWrapper(zarr_array))
        try_nczarr = self._mode == 'r'
        dimensions, attributes = _get_zarr_dims_and_attrs(zarr_array, DIMENSION_KEY, try_nczarr)
        attributes = dict(attributes)
        attributes.pop('filters', None)
        encoding = {'chunks': zarr_array.chunks, 'preferred_chunks': dict(zip(dimensions, zarr_array.chunks)), 'compressor': zarr_array.compressor, 'filters': zarr_array.filters}
        if getattr(zarr_array, 'fill_value') is not None:
            attributes['_FillValue'] = zarr_array.fill_value
        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenDict(((k, self.open_store_variable(k, v)) for k, v in self.zarr_group.arrays()))

    def get_attrs(self):
        return {k: v for k, v in self.zarr_group.attrs.asdict().items() if not k.lower().startswith('_nc')}

    def get_dimensions(self):
        try_nczarr = self._mode == 'r'
        dimensions = {}
        for k, v in self.zarr_group.arrays():
            dim_names, _ = _get_zarr_dims_and_attrs(v, DIMENSION_KEY, try_nczarr)
            for d, s in zip(dim_names, v.shape):
                if d in dimensions and dimensions[d] != s:
                    raise ValueError(f'found conflicting lengths for dimension {d} ({s} != {dimensions[d]})')
                dimensions[d] = s
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if unlimited_dims is not None:
            raise NotImplementedError("Zarr backend doesn't know how to handle unlimited dimensions")

    def set_attributes(self, attributes):
        _put_attrs(self.zarr_group, attributes)

    def encode_variable(self, variable):
        variable = encode_zarr_variable(variable)
        return variable

    def encode_attribute(self, a):
        return encode_zarr_attr_value(a)

    def store(self, variables, attributes, check_encoding_set=frozenset(), writer=None, unlimited_dims=None):
        """
        Top level method for putting data on this store, this method:
          - encodes variables/attributes
          - sets dimensions
          - sets variables

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
            dimension on which the zarray will be appended
            only needed in append mode
        """
        import zarr
        existing_keys = tuple(self.zarr_group.array_keys())
        existing_variable_names = {vn for vn in variables if _encode_variable_name(vn) in existing_keys}
        new_variables = set(variables) - existing_variable_names
        variables_without_encoding = {vn: variables[vn] for vn in new_variables}
        variables_encoded, attributes = self.encode(variables_without_encoding, attributes)
        if existing_variable_names:
            existing_vars, _, _ = conventions.decode_cf_variables({k: v for k, v in self.get_variables().items() if k in existing_variable_names}, self.get_attrs())
            vars_with_encoding = {}
            for vn in existing_variable_names:
                vars_with_encoding[vn] = variables[vn].copy(deep=False)
                vars_with_encoding[vn].encoding = existing_vars[vn].encoding
            vars_with_encoding, _ = self.encode(vars_with_encoding, {})
            variables_encoded.update(vars_with_encoding)
            for var_name in existing_variable_names:
                variables_encoded[var_name] = _validate_and_transpose_existing_dims(var_name, variables_encoded[var_name], existing_vars[var_name], self._write_region, self._append_dim)
        if self._mode not in ['r', 'r+']:
            self.set_attributes(attributes)
            self.set_dimensions(variables_encoded, unlimited_dims=unlimited_dims)
        if self._mode == 'a-' and self._append_dim is not None:
            variables_to_set = {k: v for k, v in variables_encoded.items() if k not in existing_variable_names or self._append_dim in v.dims}
        else:
            variables_to_set = variables_encoded
        self.set_variables(variables_to_set, check_encoding_set, writer, unlimited_dims=unlimited_dims)
        if self._consolidate_on_close:
            zarr.consolidate_metadata(self.zarr_group.store)

    def sync(self):
        pass

    def set_variables(self, variables, check_encoding_set, writer, unlimited_dims=None):
        """
        This provides a centralized method to set the variables on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        import zarr
        existing_keys = tuple(self.zarr_group.array_keys())
        for vn, v in variables.items():
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            attrs = v.attrs.copy()
            dims = v.dims
            dtype = v.dtype
            shape = v.shape
            fill_value = attrs.pop('_FillValue', None)
            if v.encoding == {'_FillValue': None} and fill_value is None:
                v.encoding = {}
            encoding = extract_zarr_variable_encoding(v, raise_on_invalid=check, name=vn, safe_chunks=self._safe_chunks)
            if name in existing_keys:
                if self._write_empty is not None:
                    zarr_array = zarr.open(store=self.zarr_group.chunk_store, path=f'{self.zarr_group.name}/{name}', write_empty_chunks=self._write_empty)
                else:
                    zarr_array = self.zarr_group[name]
            else:
                encoded_attrs = {}
                encoded_attrs[DIMENSION_KEY] = dims
                for k2, v2 in attrs.items():
                    encoded_attrs[k2] = self.encode_attribute(v2)
                if coding.strings.check_vlen_dtype(dtype) == str:
                    dtype = str
                if self._write_empty is not None:
                    if 'write_empty_chunks' in encoding and encoding['write_empty_chunks'] != self._write_empty:
                        raise ValueError(f'Differing "write_empty_chunks" values in encoding and parametersGot encoding["write_empty_chunks"] = {encoding['write_empty_chunks']!r} and self._write_empty = {self._write_empty!r}')
                    else:
                        encoding['write_empty_chunks'] = self._write_empty
                zarr_array = self.zarr_group.create(name, shape=shape, dtype=dtype, fill_value=fill_value, **encoding)
                zarr_array = _put_attrs(zarr_array, encoded_attrs)
            write_region = self._write_region if self._write_region is not None else {}
            write_region = {dim: write_region.get(dim, slice(None)) for dim in dims}
            if self._append_dim is not None and self._append_dim in dims:
                append_axis = dims.index(self._append_dim)
                assert write_region[self._append_dim] == slice(None)
                write_region[self._append_dim] = slice(zarr_array.shape[append_axis], None)
                new_shape = list(zarr_array.shape)
                new_shape[append_axis] += v.shape[append_axis]
                zarr_array.resize(new_shape)
            region = tuple((write_region[dim] for dim in dims))
            writer.add(v.data, zarr_array, region)

    def close(self):
        if self._close_store_on_close:
            self.zarr_group.store.close()
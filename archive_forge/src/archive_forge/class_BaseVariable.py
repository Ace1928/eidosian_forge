import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
class BaseVariable:

    def __init__(self, parent, name, dimensions=None):
        self._parent_ref = weakref.ref(parent)
        self._root_ref = weakref.ref(parent._root)
        self._h5path = _join_h5paths(parent.name, name)
        self._dimensions = dimensions
        self._initialized = True

    @property
    def _parent(self):
        return self._parent_ref()

    @property
    def _root(self):
        return self._root_ref()

    @property
    def _h5ds(self):
        return self._root._h5file[self._h5path]

    @property
    def name(self):
        """Return variable name."""
        return self._h5ds.name.replace('_nc4_non_coord_', '')

    def _lookup_dimensions(self):
        attrs = self._h5ds.attrs
        if '_Netcdf4Coordinates' in attrs and attrs.get('CLASS', None) == b'DIMENSION_SCALE':
            order_dim = {value._dimid: key for key, value in self._parent._all_dimensions.items()}
            return tuple((order_dim[coord_id] for coord_id in attrs['_Netcdf4Coordinates']))
        if 'DIMENSION_LIST' in attrs:
            if _unlabeled_dimension_mix(self._h5ds) == 'labeled':
                return tuple((self._root._h5file[ref[-1]].name.split('/')[-1] for ref in list(self._h5ds.attrs.get('DIMENSION_LIST', []))))
        child_name = self._h5ds.name.split('/')[-1]
        if child_name in self._parent._all_dimensions:
            return (child_name,)
        dims = []
        phony_dims = defaultdict(int)
        for axis, dim in enumerate(self._h5ds.dims):
            if len(dim):
                name = _name_from_dimension(dim)
            elif self._root._phony_dims_mode is None:
                raise ValueError(f"variable {self.name!r} has no dimension scale associated with axis {axis}. \nUse phony_dims='sort' for sorted naming or phony_dims='access' for per access naming.")
            else:
                dimsize = self._h5ds.shape[axis]
                dim_names = [d.name for d in self._parent._all_dimensions.maps[0].values() if d.size == dimsize]
                name = dim_names[phony_dims[dimsize]].split('/')[-1]
                phony_dims[dimsize] += 1
            dims.append(name)
        return tuple(dims)

    def _attach_dim_scales(self):
        """Attach dimension scales"""
        for n, dim in enumerate(self.dimensions):
            self._h5ds.dims[n].attach_scale(self._parent._all_dimensions[dim]._h5ds)

    def _attach_coords(self):
        dims = self.dimensions
        coord_ids = np.array([self._parent._all_dimensions[d]._dimid for d in dims], 'int32')
        if len(coord_ids) > 1:
            self._h5ds.attrs['_Netcdf4Coordinates'] = coord_ids

    def _ensure_dim_id(self):
        """Set _Netcdf4Dimid"""
        if self.dimensions and (not self._h5ds.attrs.get('_Netcdf4Dimid', False)):
            dim = self._parent._all_h5groups[self.dimensions[0]]
            if '_Netcdf4Dimid' in dim.attrs:
                self._h5ds.attrs['_Netcdf4Dimid'] = dim.attrs['_Netcdf4Dimid']

    def _maybe_resize_dimensions(self, key, value):
        """Resize according to given (expanded) key with respect to variable dimensions"""
        new_shape = ()
        v = None
        for i, dim in enumerate(self.dimensions):
            if self._parent._all_dimensions[dim].isunlimited():
                if key[i].stop is None:
                    if v is None:
                        v = np.asarray(value)
                    if v.ndim == self.ndim:
                        new_max = max(v.shape[i], self._h5ds.shape[i])
                    elif v.ndim == 0:
                        new_max = self._parent._all_dimensions[dim].size
                    else:
                        raise IndexError('shape of data does not conform to slice')
                else:
                    new_max = max(key[i].stop, self._h5ds.shape[i])
                if self._parent._all_dimensions[dim].size < new_max:
                    self._parent.resize_dimension(dim, new_max)
                new_shape += (new_max,)
            else:
                new_shape += (self._parent._all_dimensions[dim].size,)
        if self._h5ds.shape != new_shape:
            self._h5ds.resize(new_shape)

    @property
    def dimensions(self):
        """Return variable dimension names."""
        if self._dimensions is None:
            self._dimensions = self._lookup_dimensions()
        return self._dimensions

    @property
    def shape(self):
        """Return current sizes of all variable dimensions."""
        return tuple([self._parent._all_dimensions[d].size for d in self.dimensions])

    @property
    def ndim(self):
        """Return number variable dimensions"""
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        """Return NumPy dtype object giving the variable’s type."""
        return self._h5ds.dtype

    def _get_padding(self, key):
        """Return padding if needed, defaults to False."""
        padding = False
        if self.dtype != str and self.dtype.kind in ['f', 'i', 'u']:
            key0 = _expanded_indexer(key, self.ndim)
            key0 = _transform_1d_boolean_indexers(key0)
            h5ds_shape = self._h5ds.shape
            shape = self.shape
            max_index = [max(k) + 1 if isinstance(k, (np.ndarray, list)) else k.stop for k in key0]
            max_shape = tuple([shape[i] if k is None else max(h5ds_shape[i], k) for i, k in enumerate(max_index)])
            sdiff = [d0 - d1 for d0, d1 in zip(max_shape, h5ds_shape)]
            if sum(sdiff):
                padding = [(0, s) for s in sdiff]
        return padding

    def __array__(self, *args, **kwargs):
        return self._h5ds.__array__(*args, **kwargs)

    def __getitem__(self, key):
        from .legacyapi import Dataset
        if isinstance(self._parent._root, Dataset):
            h5py_version = version.parse(h5py.__version__)
            if version.parse('3.0.0') <= h5py_version < version.parse('3.7.0'):
                key = _transform_1d_boolean_indexers(key)
        if getattr(self._root, 'decode_vlen_strings', False):
            string_info = self._root._h5py.check_string_dtype(self._h5ds.dtype)
            if string_info and string_info.length is None:
                return self._h5ds.asstr()[key]
        padding = self._get_padding(key)
        if padding:
            fv = self.dtype.type(self._h5ds.fillvalue)
            return np.pad(self._h5ds, pad_width=padding, mode='constant', constant_values=fv)[key]
        return self._h5ds[key]

    def __setitem__(self, key, value):
        from .legacyapi import Dataset
        if isinstance(self._parent._root, Dataset):
            key = _expanded_indexer(key, self.ndim)
            key = _transform_1d_boolean_indexers(key)
            self._maybe_resize_dimensions(key, value)
        self._h5ds[key] = value

    @property
    def attrs(self):
        """Return variable attributes."""
        return Attributes(self._h5ds.attrs, self._root._check_valid_netcdf_dtype, self._root._h5py)
    _cls_name = 'h5netcdf.Variable'

    def __repr__(self):
        if self._parent._root._closed:
            return '<Closed %s>' % self._cls_name
        header = '<{} {!r}: dimensions {}, shape {}, dtype {}>'.format(self._cls_name, self.name, self.dimensions, self.shape, self.dtype)
        return '\n'.join([header] + ['Attributes:'] + [f'    {k}: {v!r}' for k, v in self.attrs.items()])
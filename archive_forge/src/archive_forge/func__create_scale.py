import weakref
from collections import OrderedDict
from collections.abc import MutableMapping
import h5py
import numpy as np
def _create_scale(self):
    """Create dimension scale for this dimension"""
    if self._name not in self._parent._h5group:
        kwargs = {}
        if self._size is None or self._size == 0:
            kwargs['maxshape'] = (None,)
        if self._root._h5py.__name__ == 'h5py':
            kwargs.update(dict(track_order=self._parent._track_order))
        self._parent._h5group.create_dataset(name=self._name, shape=(self._size,), dtype='>f4', **kwargs)
    self._h5ds.attrs['_Netcdf4Dimid'] = np.array(self._dimid, dtype=np.int32)
    if len(self._h5ds.shape) > 1:
        dims = self._parent._variables[self._name].dimensions
        coord_ids = np.array([self._parent._dimensions[d]._dimid for d in dims], 'int32')
        self._h5ds.attrs['_Netcdf4Coordinates'] = coord_ids
    size = self._size
    if not size:
        size = 1
    if isinstance(size, tuple):
        size = size[0]
    dimlen = bytes(f'{size:10}', 'ascii')
    NOT_A_VARIABLE = b'This is a netCDF dimension but not a netCDF variable.'
    scale_name = self.name if self.name in self._parent._variables else NOT_A_VARIABLE + dimlen
    if not self._root._h5py.h5ds.is_scale(self._h5ds.id):
        self._h5ds.make_scale(scale_name)
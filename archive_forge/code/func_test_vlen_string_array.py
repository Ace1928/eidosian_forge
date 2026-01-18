import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_vlen_string_array(self):
    """ Storage of vlen byte string arrays"""
    dt = h5py.string_dtype(encoding='ascii')
    data = np.ndarray((2,), dtype=dt)
    data[...] = ('Hello', 'Hi there!  This is HDF5!')
    self.f.attrs['x'] = data
    out = self.f.attrs['x']
    self.assertEqual(out.dtype, dt)
    self.assertEqual(out[0], data[0])
    self.assertEqual(out[1], data[1])
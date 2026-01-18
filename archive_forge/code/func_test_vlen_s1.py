import numpy as np
from collections.abc import MutableMapping
from .common import TestCase, ut
import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager
def test_vlen_s1(self):
    dt = h5py.vlen_dtype(np.dtype('S1'))
    a = np.empty((1,), dtype=dt)
    a[0] = np.array([b'a', b'b'], dtype='S1')
    self.f.attrs.create('test', a)
    self.assertArrayEqual(self.f.attrs['test'][0], a[0])
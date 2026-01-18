import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_reference_field(self):
    """ Compound types of which a reference is an element work right """
    dt = np.dtype([('a', 'i'), ('b', h5py.ref_dtype)])
    dset = self.f.create_dataset('x', (1,), dtype=dt)
    dset[0] = (42, self.f['/'].ref)
    out = dset[0]
    self.assertEqual(type(out[1]), h5py.Reference)
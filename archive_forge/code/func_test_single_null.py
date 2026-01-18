import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_single_null(self):
    """ Single-element selection with [()] yields ndarray """
    dset = self.f.create_dataset('x', (1,), dtype='i1')
    out = dset[()]
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, (1,))
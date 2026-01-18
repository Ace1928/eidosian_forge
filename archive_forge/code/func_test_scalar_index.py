import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_scalar_index(self):
    """ Slicing with [...] yields scalar ndarray """
    dset = self.f.create_dataset('x', shape=(), dtype='f')
    out = dset[...]
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, ())
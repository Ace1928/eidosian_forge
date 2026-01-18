import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_single_index(self):
    """ Single-element selection with [index] yields array scalar """
    dset = self.f.create_dataset('x', (1,), dtype='i1')
    out = dset[0]
    self.assertIsInstance(out, np.int8)
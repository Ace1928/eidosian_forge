import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_write_noncompound(self):
    """ Test write with non-compound source (single-field) """
    data2 = self.data.copy()
    data2['b'] = 1.0
    self.dset['b'] = 1.0
    self.assertTrue(np.all(self.dset[...] == data2))
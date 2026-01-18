import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_write_element(self):
    """ Write a single element to the array

        Issue 211.
        """
    dt = np.dtype('(3,)f8')
    dset = self.f.create_dataset('x', (10,), dtype=dt)
    data = np.array([1, 2, 3.0])
    dset[4] = data
    out = dset[4]
    self.assertTrue(np.all(out == data))
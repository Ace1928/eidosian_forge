import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_write_slices(self):
    """ Write slices to array type """
    dt = np.dtype('(3,)i')
    data1 = np.ones((2,), dtype=dt)
    data2 = np.ones((4, 5), dtype=dt)
    dset = self.f.create_dataset('x', (10, 9, 11), dtype=dt)
    dset[0, 0, 2:4] = data1
    self.assertArrayEqual(dset[0, 0, 2:4], data1)
    dset[3, 1:5, 6:11] = data2
    self.assertArrayEqual(dset[3, 1:5, 6:11], data2)
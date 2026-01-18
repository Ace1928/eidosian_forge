import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_write_broadcast(self):
    """ Array fill from constant is not supported (issue 211).
        """
    dt = np.dtype('(3,)i')
    dset = self.f.create_dataset('x', (10,), dtype=dt)
    with self.assertRaises(TypeError):
        dset[...] = 42
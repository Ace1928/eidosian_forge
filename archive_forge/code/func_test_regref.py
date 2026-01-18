import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_regref(self):
    """ Indexing a region reference dataset returns a h5py.RegionReference
        """
    dset1 = self.f.create_dataset('x', (10, 10))
    regref = dset1.regionref[...]
    dset2 = self.f.create_dataset('y', (1,), dtype=h5py.regionref_dtype)
    dset2[0] = regref
    self.assertEqual(type(dset2[0]), h5py.RegionReference)
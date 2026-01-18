import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_slice_of_length_zero(self):
    """ Get a slice of length zero from a non-empty dataset """
    for i, shape in enumerate([(3,), (2, 2), (2, 1, 5)]):
        dset = self.f.create_dataset('x%d' % i, data=np.zeros(shape, int), maxshape=(None,) * len(shape))
        self.assertEqual(dset.shape, shape)
        out = dset[1:1]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (0,) + shape[1:])
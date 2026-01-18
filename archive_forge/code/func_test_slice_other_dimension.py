import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_slice_other_dimension(self):
    """ Slice a dataset with a zero in its shape vector
            along a non-zero-length dimension """
    for i, shape in enumerate([(3, 0), (1, 2, 0), (2, 0, 1)]):
        dset = self.f.create_dataset('x%d' % i, shape, dtype=int, maxshape=(None,) * len(shape))
        self.assertEqual(dset.shape, shape)
        out = dset[:1]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,) + shape[1:])
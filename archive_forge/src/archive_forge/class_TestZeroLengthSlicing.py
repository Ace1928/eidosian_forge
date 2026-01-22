import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestZeroLengthSlicing(BaseSlicing):
    """
        Slices resulting in empty arrays
    """

    def test_slice_zero_length_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along the zero-length dimension """
        for i, shape in enumerate([(0,), (0, 3), (0, 2, 1)]):
            dset = self.f.create_dataset('x%d' % i, shape, dtype=int, maxshape=(None,) * len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[...]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            out = dset[:]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            if len(shape) > 1:
                out = dset[:, :1]
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape[:2], (0, 1))

    def test_slice_other_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along a non-zero-length dimension """
        for i, shape in enumerate([(3, 0), (1, 2, 0), (2, 0, 1)]):
            dset = self.f.create_dataset('x%d' % i, shape, dtype=int, maxshape=(None,) * len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (1,) + shape[1:])

    def test_slice_of_length_zero(self):
        """ Get a slice of length zero from a non-empty dataset """
        for i, shape in enumerate([(3,), (2, 2), (2, 1, 5)]):
            dset = self.f.create_dataset('x%d' % i, data=np.zeros(shape, int), maxshape=(None,) * len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[1:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (0,) + shape[1:])
import sys
import numpy as np
import h5py
from .common import ut, TestCase
class TestScalarFloat(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.data = np.array(42.5, dtype=np.double)
        self.dset = self.f.create_dataset('x', data=self.data)

    def test_ndim(self):
        """ Verify number of dimensions """
        self.assertEqual(self.dset.ndim, 0)

    def test_size(self):
        """ Verify size """
        self.assertEqual(self.dset.size, 1)

    def test_nbytes(self):
        """ Verify nbytes """
        self.assertEqual(self.dset.nbytes, self.data.dtype.itemsize)

    def test_shape(self):
        """ Verify shape """
        self.assertEqual(self.dset.shape, tuple())

    def test_ellipsis(self):
        """ Ellipsis -> scalar ndarray """
        out = self.dset[...]
        self.assertArrayEqual(out, self.data)

    def test_tuple(self):
        """ () -> bare item """
        out = self.dset[()]
        self.assertArrayEqual(out, self.data.item())

    def test_slice(self):
        """ slice -> ValueError """
        with self.assertRaises(ValueError):
            self.dset[0:4]

    def test_multi_block_slice(self):
        """ MultiBlockSlice -> ValueError """
        with self.assertRaises(ValueError):
            self.dset[h5py.MultiBlockSlice()]

    def test_index(self):
        """ index -> ValueError """
        with self.assertRaises(ValueError):
            self.dset[0]

    def test_indexlist(self):
        """ index list -> ValueError """
        with self.assertRaises(ValueError):
            self.dset[[1, 2, 5]]

    def test_mask(self):
        """ mask -> ValueError """
        mask = np.array(True, dtype='bool')
        with self.assertRaises(ValueError):
            self.dset[mask]

    def test_fieldnames(self):
        """ field name -> ValueError (no fields) """
        with self.assertRaises(ValueError):
            self.dset['field']
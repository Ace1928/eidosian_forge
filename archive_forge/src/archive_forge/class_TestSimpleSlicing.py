import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
class TestSimpleSlicing(TestCase):
    """
        Feature: Simple NumPy-style slices (start:stop:step) are supported.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_negative_stop(self):
        """ Negative stop indexes work as they do in NumPy """
        self.assertArrayEqual(self.dset[2:-2], self.arr[2:-2])

    def test_write(self):
        """Assigning to a 1D slice of a 2D dataset
        """
        dset = self.f.create_dataset('x2', (10, 2))
        x = np.zeros((10, 1))
        dset[:, 0] = x[:, 0]
        with self.assertRaises(TypeError):
            dset[:, 1] = x
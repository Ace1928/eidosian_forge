import sys
import numpy as np
import h5py
from .common import ut, TestCase
class TestBoolIndex(TestCase):
    """
    Tests for indexing with Boolean arrays
    """

    def setUp(self):
        super().setUp()
        self.arr = np.arange(9).reshape(3, -1)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def test_select_first_axis(self):
        sel = np.s_[[False, True, False], :]
        self.assertNumpyBehavior(self.dset, self.arr, sel)

    def test_wrong_size(self):
        sel = np.s_[[False, True, False, False], :]
        with self.assertRaises(TypeError):
            self.dset[sel]
import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_slice_backwards(self):
    """ we disallow negative steps """
    with self.assertRaises(ValueError):
        self.dset[::-1]
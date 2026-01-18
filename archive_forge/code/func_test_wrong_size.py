import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_wrong_size(self):
    sel = np.s_[[False, True, False, False], :]
    with self.assertRaises(TypeError):
        self.dset[sel]
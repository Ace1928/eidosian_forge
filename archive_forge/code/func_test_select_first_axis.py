import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_select_first_axis(self):
    sel = np.s_[[False, True, False], :]
    self.assertNumpyBehavior(self.dset, self.arr, sel)
import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_slice_stop_less_than_start(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[7:5])
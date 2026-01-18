import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_slice_strides(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[1:7:3])
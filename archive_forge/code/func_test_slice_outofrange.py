import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_slice_outofrange(self):
    self.assertNumpyBehavior(self.dset, self.data, np.s_[100:400:3])
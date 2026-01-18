import numpy as np
import h5py
from .common import ut, TestCase
def test_switch_swmr_mode_on(self):
    """ Switch to SWMR mode and verify """
    self.f.swmr_mode = True
    self.assertTrue(self.f.swmr_mode)
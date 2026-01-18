import numpy as np
import h5py
from .common import ut, TestCase
def test_switch_swmr_mode_off_raises(self):
    """ Switching SWMR write mode off is only possible by closing the file.
        Attempts to forcibly switch off the SWMR mode should raise a ValueError.
        """
    self.f.swmr_mode = True
    self.assertTrue(self.f.swmr_mode)
    with self.assertRaises(ValueError):
        self.f.swmr_mode = False
    self.assertTrue(self.f.swmr_mode)
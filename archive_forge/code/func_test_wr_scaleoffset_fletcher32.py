import os
import numpy as np
import h5py
from .common import ut, TestCase
def test_wr_scaleoffset_fletcher32(self):
    """ make sure that scaleoffset + fletcher32 is prevented
        """
    data = np.linspace(0, 1, 100)
    with self.assertRaises(ValueError):
        self.f.create_dataset('test_data', data=data, fletcher32=True, scaleoffset=3)
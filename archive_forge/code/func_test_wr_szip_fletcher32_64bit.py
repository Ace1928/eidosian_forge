import os
import numpy as np
import h5py
from .common import ut, TestCase
@ut.skipUnless(h5py.h5z.filter_avail(h5py.h5z.FILTER_SZIP), 'szip filter required')
def test_wr_szip_fletcher32_64bit(self):
    """ test combination of szip, fletcher32, and 64bit arrays

        The fletcher32 checksum must be computed after the szip
        compression is applied.

        References:
        - GitHub issue #953
        - https://lists.hdfgroup.org/pipermail/
          hdf-forum_lists.hdfgroup.org/2018-January/010753.html
        """
    self.f.create_dataset('test_data', data=np.zeros(10000, dtype=np.float64), fletcher32=True, compression='szip')
    self.f.close()
    with h5py.File(self.path, 'r') as h5:
        h5['test_data'][0]
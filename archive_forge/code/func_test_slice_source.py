import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
def test_slice_source(self):
    outfile = self.make_virtual_ds()
    with h5.File(outfile, 'r') as f:
        assert_array_equal(f['/group/data'][0][:3], [1, 3, 5])
        assert_array_equal(f['/group/data'][0][50:53], [2, 4, 6])
        assert_array_equal(f['/group/data'][3][:3], [4, 6, 8])
        assert_array_equal(f['/group/data'][3][50:53], [5, 7, 9])
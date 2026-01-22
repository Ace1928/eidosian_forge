import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
@ut.skipUnless(vds_support, 'VDS requires HDF5 >= 1.9.233')
class RelativeLinkTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.f1 = osp.join(self.tmpdir, 'testfile1.h5')
        self.f2 = osp.join(self.tmpdir, 'testfile2.h5')
        self.data1 = np.arange(10)
        self.data2 = np.arange(10) * -1
        with h5.File(self.f1, 'w') as f:
            ds = f.create_dataset('data', (10,), 'f4')
            ds[:] = self.data1
        with h5.File(self.f2, 'w') as f:
            ds = f.create_dataset('data', (10,), 'f4')
            ds[:] = self.data2
            self.make_vds(f)

    def make_vds(self, f):
        layout = h5.VirtualLayout((2, 10), 'f4')
        vsource1 = h5.VirtualSource(self.f1, 'data', shape=(10,))
        vsource2 = h5.VirtualSource(self.f2, 'data', shape=(10,))
        layout[0] = vsource1
        layout[1] = vsource2
        f.create_virtual_dataset('virtual', layout)

    def test_relative_vds(self):
        with h5.File(self.f2) as f:
            data = f['virtual'][:]
            np.testing.assert_array_equal(data[0], self.data1)
            np.testing.assert_array_equal(data[1], self.data2)
        f3 = osp.join(self.tmpdir, 'testfile3.h5')
        os.rename(self.f2, f3)
        with h5.File(f3) as f:
            data = f['virtual'][:]
            assert data.dtype == 'f4'
            np.testing.assert_array_equal(data[0], self.data1)
            np.testing.assert_array_equal(data[1], self.data2)
        f4 = osp.join(self.tmpdir, 'testfile4.h5')
        os.rename(self.f1, f4)
        with h5.File(f3) as f:
            data = f['virtual'][:]
            assert data.dtype == 'f4'
            np.testing.assert_array_equal(data[0], 0)
            np.testing.assert_array_equal(data[1], self.data2)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
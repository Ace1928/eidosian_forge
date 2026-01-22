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
class SlicingTestCase(ut.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for n in range(1, 5):
            with h5.File(osp.join(self.tmpdir, '{}.h5'.format(n)), 'w') as f:
                d = f.create_dataset('data', (100,), 'i4')
                d[:] = np.arange(100) + n

    def make_virtual_ds(self):
        layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))
        for n in range(1, 5):
            filename = osp.join(self.tmpdir, '{}.h5'.format(n))
            vsource = h5.VirtualSource(filename, 'data', shape=(100,))
            layout[n - 1, :50] = vsource[0:100:2]
            layout[n - 1, 50:] = vsource[1:100:2]
        outfile = osp.join(self.tmpdir, 'VDS.h5')
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('/group/data', layout, fillvalue=-5)
        return outfile

    def test_slice_source(self):
        outfile = self.make_virtual_ds()
        with h5.File(outfile, 'r') as f:
            assert_array_equal(f['/group/data'][0][:3], [1, 3, 5])
            assert_array_equal(f['/group/data'][0][50:53], [2, 4, 6])
            assert_array_equal(f['/group/data'][3][:3], [4, 6, 8])
            assert_array_equal(f['/group/data'][3][50:53], [5, 7, 9])

    def test_inspection(self):
        with h5.File(osp.join(self.tmpdir, '1.h5'), 'r') as f:
            assert not f['data'].is_virtual
        outfile = self.make_virtual_ds()
        with h5.File(outfile, 'r') as f:
            ds = f['/group/data']
            assert ds.is_virtual
            src_files = {osp.join(self.tmpdir, '{}.h5'.format(n)) for n in range(1, 5)}
            assert {s.file_name for s in ds.virtual_sources()} == src_files

    def test_mismatched_selections(self):
        layout = h5.VirtualLayout((4, 100), 'i4', maxshape=(4, None))
        filename = osp.join(self.tmpdir, '1.h5')
        vsource = h5.VirtualSource(filename, 'data', shape=(100,))
        with self.assertRaisesRegex(ValueError, 'different number'):
            layout[0, :49] = vsource[0:100:2]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
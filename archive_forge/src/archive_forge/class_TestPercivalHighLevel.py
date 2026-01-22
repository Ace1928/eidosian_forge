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
class TestPercivalHighLevel(ut.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['raw_file_1.h5', 'raw_file_2.h5', 'raw_file_3.h5']
        k = 0
        for outfile in self.fname:
            filename = osp.join(self.working_dir, outfile)
            f = h5.File(filename, 'w')
            f['data'] = np.ones((20, 200, 200)) * k
            k += 1
            f.close()
        f = h5.File(osp.join(self.working_dir, 'raw_file_4.h5'), 'w')
        f['data'] = np.ones((19, 200, 200)) * 3
        self.fname.append('raw_file_4.h5')
        self.fname = [osp.join(self.working_dir, ix) for ix in self.fname]
        f.close()

    def test_percival_high_level(self):
        outfile = osp.join(self.working_dir, 'percival.h5')
        layout = h5.VirtualLayout(shape=(79, 200, 200), dtype=np.float64)
        for k, filename in enumerate(self.fname):
            dim1 = 19 if k == 3 else 20
            vsource = h5.VirtualSource(filename, 'data', shape=(dim1, 200, 200))
            layout[k:79:4, :, :] = vsource[:, :, :]
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-5)
        foo = np.array(2 * list(range(4)))
        with h5.File(outfile, 'r') as f:
            ds = f['data']
            line = ds[:8, 100, 100]
            self.assertEqual(ds.shape, (79, 200, 200))
            assert_array_equal(line, foo)

    def test_percival_source_from_dataset(self):
        outfile = osp.join(self.working_dir, 'percival.h5')
        layout = h5.VirtualLayout(shape=(79, 200, 200), dtype=np.float64)
        for k, filename in enumerate(self.fname):
            with h5.File(filename, 'r') as f:
                vsource = h5.VirtualSource(f['data'])
                layout[k:79:4, :, :] = vsource
        with h5.File(outfile, 'w', libver='latest') as f:
            f.create_virtual_dataset('data', layout, fillvalue=-5)
        foo = np.array(2 * list(range(4)))
        with h5.File(outfile, 'r') as f:
            ds = f['data']
            line = ds[:8, 100, 100]
            self.assertEqual(ds.shape, (79, 200, 200))
            assert_array_equal(line, foo)

    def tearDown(self):
        shutil.rmtree(self.working_dir)
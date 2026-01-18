import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
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
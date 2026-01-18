from ..common import ut
import numpy as np
import h5py as h5
import tempfile
def test_percival_low_level(self):
    self.outfile = self.working_dir + 'percival.h5'
    with h5.File(self.outfile, 'w', libver='latest') as f:
        vdset_shape = (1, 200, 200)
        num = h5.h5s.UNLIMITED
        vdset_max_shape = (num,) + vdset_shape[1:]
        virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
        dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
        dcpl.set_fill_value(np.array([-1]))
        k = 0
        for foo in self.fname:
            in_data = h5.File(foo, 'r')['data']
            src_shape = in_data.shape
            max_src_shape = (num,) + src_shape[1:]
            in_data.file.close()
            src_dspace = h5.h5s.create_simple(src_shape, max_src_shape)
            src_dspace.select_hyperslab(start=(0, 0, 0), stride=(1, 1, 1), count=(num, 1, 1), block=(1,) + src_shape[1:])
            virt_dspace.select_hyperslab(start=(k, 0, 0), stride=(4, 1, 1), count=(num, 1, 1), block=(1,) + src_shape[1:])
            dcpl.set_virtual(virt_dspace, foo.encode('utf-8'), b'data', src_dspace)
            k += 1
        dset = h5.h5d.create(f.id, name=b'data', tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)
        f = h5.File(self.outfile, 'r')
        sh = f['data'].shape
        line = f['data'][:8, 100, 100]
        foo = np.array(2 * list(range(4)))
        f.close()
        self.assertEqual(sh, (79, 200, 200))
        np.testing.assert_array_equal(line, foo)
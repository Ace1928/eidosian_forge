from ..common import ut
import numpy as np
import h5py as h5
import tempfile
class TestExcaliburLowLevel(ut.TestCase):

    def create_excalibur_fem_stripe_datafile(self, fname, nframes, excalibur_data, scale):
        shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        max_shape = (nframes,) + excalibur_data.fem_stripe_dimensions
        chunk = (1,) + excalibur_data.fem_stripe_dimensions
        with h5.File(fname, 'w', libver='latest') as f:
            dset = f.create_dataset('data', shape=shape, maxshape=max_shape, chunks=chunk, dtype='uint16')
            for data_value_index in np.arange(nframes):
                dset[data_value_index] = excalibur_data.generate_fem_stripe_image(data_value_index * scale)

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()
        self.fname = ['stripe_%d.h5' % stripe for stripe in range(1, 7)]
        self.fname = [self.working_dir + ix for ix in self.fname]
        nframes = 5
        self.edata = ExcaliburData()
        k = 0
        for raw_file in self.fname:
            self.create_excalibur_fem_stripe_datafile(raw_file, nframes, self.edata, k)
            k += 1

    def test_excalibur_low_level(self):
        excalibur_data = self.edata
        self.outfile = self.working_dir + 'excalibur.h5'
        vdset_stripe_shape = (1,) + excalibur_data.fem_stripe_dimensions
        vdset_stripe_max_shape = (5,) + excalibur_data.fem_stripe_dimensions
        vdset_shape = (5, excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + 10 * (len(self.fname) - 1), excalibur_data.fem_stripe_dimensions[1])
        vdset_max_shape = (5, excalibur_data.fem_stripe_dimensions[0] * len(self.fname) + 10 * (len(self.fname) - 1), excalibur_data.fem_stripe_dimensions[1])
        vdset_y_offset = 0
        with h5.File(self.outfile, 'w', libver='latest') as f:
            src_dspace = h5.h5s.create_simple(vdset_stripe_shape, vdset_stripe_max_shape)
            virt_dspace = h5.h5s.create_simple(vdset_shape, vdset_max_shape)
            dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array([1]))
            src_dspace.select_hyperslab(start=(0, 0, 0), count=(1, 1, 1), block=vdset_stripe_max_shape)
            for raw_file in self.fname:
                virt_dspace.select_hyperslab(start=(0, vdset_y_offset, 0), count=(1, 1, 1), block=vdset_stripe_max_shape)
                dcpl.set_virtual(virt_dspace, raw_file.encode('utf-8'), b'/data', src_dspace)
                vdset_y_offset += vdset_stripe_shape[1] + 10
            dset = h5.h5d.create(f.id, name=b'data', tid=h5.h5t.NATIVE_INT16, space=virt_dspace, dcpl=dcpl)
            assert f['data'].fillvalue == 1
        f = h5.File(self.outfile, 'r')['data']
        self.assertEqual(f[3, 100, 0], 0.0)
        self.assertEqual(f[3, 260, 0], 1.0)
        self.assertEqual(f[3, 350, 0], 3.0)
        self.assertEqual(f[3, 650, 0], 6.0)
        self.assertEqual(f[3, 900, 0], 9.0)
        self.assertEqual(f[3, 1150, 0], 12.0)
        self.assertEqual(f[3, 1450, 0], 15.0)
        f.file.close()

    def tearDown(self):
        import os
        for f in self.fname:
            os.remove(f)
        os.remove(self.outfile)
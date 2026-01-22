from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
class TestDateTime(TestCase):
    datetime_units = ['Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']

    def test_datetime(self):
        fname = self.mktemp()
        for dt_unit in self.datetime_units:
            for dt_order in ['<', '>']:
                dt_descr = f'{dt_order}M8[{dt_unit}]'
                dt = h5py.opaque_dtype(np.dtype(dt_descr))
                arr = np.array([0], dtype=np.int64).view(dtype=dt)
                with h5py.File(fname, 'w') as f:
                    dset = f.create_dataset('default', data=arr, dtype=dt)
                    self.assertArrayEqual(arr, dset)
                    self.assertEqual(arr.dtype, dset.dtype)

    def test_timedelta(self):
        fname = self.mktemp()
        for dt_unit in self.datetime_units:
            for dt_order in ['<', '>']:
                dt_descr = f'{dt_order}m8[{dt_unit}]'
                dt = h5py.opaque_dtype(np.dtype(dt_descr))
                arr = np.array([np.timedelta64(500, dt_unit)], dtype=dt)
                with h5py.File(fname, 'w') as f:
                    dset = f.create_dataset('default', data=arr, dtype=dt)
                    self.assertArrayEqual(arr, dset)
                    self.assertEqual(arr.dtype, dset.dtype)
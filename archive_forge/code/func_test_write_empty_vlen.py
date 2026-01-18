from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_write_empty_vlen(self):
    fname = self.mktemp()
    with h5py.File(fname, 'w') as f:
        d = np.core.records.fromarrays([[], []], names='a,b', formats='|V16,O')
        dset = f.create_dataset('test', data=d, dtype=[('a', '|V16'), ('b', h5py.special_dtype(vlen=np.float_))])
        self.assertEqual(dset.size, 0)
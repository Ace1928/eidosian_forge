from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_out_of_order_offsets(self):
    dt = np.dtype({'names': ['f1', 'f2', 'f3'], 'formats': ['<f4', '<i4', '<f8'], 'offsets': [0, 16, 8]})
    data = np.zeros(10, dtype=dt)
    data['f1'] = np.random.rand(data.size)
    data['f2'] = np.random.randint(-10, 11, data.size)
    data['f3'] = np.random.rand(data.size) * -1
    fname = self.mktemp()
    with h5py.File(fname, 'w') as fd:
        fd.create_dataset('data', data=data)
    with h5py.File(fname, 'r') as fd:
        self.assertArrayEqual(fd['data'], data)
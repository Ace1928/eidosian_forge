import sys
import numpy as np
import h5py
from .common import ut, TestCase
def test_read_no_fill_value(writable_file):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((1,))
    dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    ds = h5py.Dataset(h5py.h5d.create(writable_file.id, b'a', h5py.h5t.IEEE_F64LE, h5py.h5s.create_simple((5,)), dcpl))
    np.testing.assert_array_equal(ds[:3], np.zeros(3, np.float64))
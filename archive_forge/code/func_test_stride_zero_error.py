import numpy as np
from .common import ut, TestCase
import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice
def test_stride_zero_error(self):
    with self.assertRaises(ValueError):
        MultiBlockSlice(stride=0, block=0).indices(10)
import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
class TestFloats(BaseDataset):
    """
        Test support for mini and extended-precision floats
    """

    def _exectest(self, dt):
        dset = self.f.create_dataset('x', (100,), dtype=dt)
        self.assertEqual(dset.dtype, dt)
        data = np.ones((100,), dtype=dt)
        dset[...] = data
        self.assertArrayEqual(dset[...], data)

    @ut.skipUnless(hasattr(np, 'float16'), 'NumPy float16 support required')
    def test_mini(self):
        """ Mini-floats round trip """
        self._exectest(np.dtype('float16'))

    def test_mini_mapping(self):
        """ Test mapping for float16 """
        if hasattr(np, 'float16'):
            self.assertEqual(h5t.IEEE_F16LE.dtype, np.dtype('<f2'))
        else:
            self.assertEqual(h5t.IEEE_F16LE.dtype, np.dtype('<f4'))
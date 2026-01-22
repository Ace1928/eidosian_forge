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
class TestZeroShape(BaseDataset):
    """
        Features of datasets with (0,)-shape axes
    """

    def test_array_conversion(self):
        """ Empty datasets can be converted to NumPy arrays """
        ds = self.f.create_dataset('x', 0, maxshape=None)
        self.assertEqual(ds.shape, np.array(ds).shape)
        ds = self.f.create_dataset('y', (0,), maxshape=(None,))
        self.assertEqual(ds.shape, np.array(ds).shape)
        ds = self.f.create_dataset('z', (0, 0), maxshape=(None, None))
        self.assertEqual(ds.shape, np.array(ds).shape)

    def test_reading(self):
        """ Slicing into empty datasets works correctly """
        dt = [('a', 'f'), ('b', 'i')]
        ds = self.f.create_dataset('x', (0,), dtype=dt, maxshape=(None,))
        arr = np.empty((0,), dtype=dt)
        self.assertEqual(ds[...].shape, arr.shape)
        self.assertEqual(ds[...].dtype, arr.dtype)
        self.assertEqual(ds[()].shape, arr.shape)
        self.assertEqual(ds[()].dtype, arr.dtype)
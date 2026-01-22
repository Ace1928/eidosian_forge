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
class TestChunkIterator(BaseDataset):

    def test_no_chunks(self):
        dset = self.f.create_dataset('foo', ())
        with self.assertRaises(TypeError):
            dset.iter_chunks()

    def test_1d(self):
        dset = self.f.create_dataset('foo', (100,), chunks=(32,))
        expected = ((slice(0, 32, 1),), (slice(32, 64, 1),), (slice(64, 96, 1),), (slice(96, 100, 1),))
        self.assertEqual(list(dset.iter_chunks()), list(expected))
        expected = ((slice(50, 64, 1),), (slice(64, 96, 1),), (slice(96, 97, 1),))
        self.assertEqual(list(dset.iter_chunks(np.s_[50:97])), list(expected))

    def test_2d(self):
        dset = self.f.create_dataset('foo', (100, 100), chunks=(32, 64))
        expected = ((slice(0, 32, 1), slice(0, 64, 1)), (slice(0, 32, 1), slice(64, 100, 1)), (slice(32, 64, 1), slice(0, 64, 1)), (slice(32, 64, 1), slice(64, 100, 1)), (slice(64, 96, 1), slice(0, 64, 1)), (slice(64, 96, 1), slice(64, 100, 1)), (slice(96, 100, 1), slice(0, 64, 1)), (slice(96, 100, 1), slice(64, 100, 1)))
        self.assertEqual(list(dset.iter_chunks()), list(expected))
        expected = ((slice(48, 52, 1), slice(40, 50, 1)),)
        self.assertEqual(list(dset.iter_chunks(np.s_[48:52, 40:50])), list(expected))
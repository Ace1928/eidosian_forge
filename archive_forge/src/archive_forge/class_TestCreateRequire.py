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
class TestCreateRequire(BaseDataset):
    """
        Feature: Datasets can be created only if they don't exist in the file
    """

    def test_create(self):
        """ Create new dataset with no conflicts """
        dset = self.f.require_dataset('foo', (10, 3), 'f')
        self.assertIsInstance(dset, Dataset)
        self.assertEqual(dset.shape, (10, 3))

    def test_create_existing(self):
        """ require_dataset yields existing dataset """
        dset = self.f.require_dataset('foo', (10, 3), 'f')
        dset2 = self.f.require_dataset('foo', (10, 3), 'f')
        self.assertEqual(dset, dset2)

    def test_create_1D(self):
        """ require_dataset with integer shape yields existing dataset"""
        dset = self.f.require_dataset('foo', 10, 'f')
        dset2 = self.f.require_dataset('foo', 10, 'f')
        self.assertEqual(dset, dset2)
        dset = self.f.require_dataset('bar', (10,), 'f')
        dset2 = self.f.require_dataset('bar', 10, 'f')
        self.assertEqual(dset, dset2)
        dset = self.f.require_dataset('baz', 10, 'f')
        dset2 = self.f.require_dataset(b'baz', (10,), 'f')
        self.assertEqual(dset, dset2)

    def test_shape_conflict(self):
        """ require_dataset with shape conflict yields TypeError """
        self.f.create_dataset('foo', (10, 3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 4), 'f')

    def test_type_conflict(self):
        """ require_dataset with object type conflict yields TypeError """
        self.f.create_group('foo')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 3), 'f')

    def test_dtype_conflict(self):
        """ require_dataset with dtype conflict (strict mode) yields TypeError
        """
        dset = self.f.create_dataset('foo', (10, 3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 3), 'S10')

    def test_dtype_exact(self):
        """ require_dataset with exactly dtype match """
        dset = self.f.create_dataset('foo', (10, 3), 'f')
        dset2 = self.f.require_dataset('foo', (10, 3), 'f', exact=True)
        self.assertEqual(dset, dset2)

    def test_dtype_close(self):
        """ require_dataset with convertible type succeeds (non-strict mode)
        """
        dset = self.f.create_dataset('foo', (10, 3), 'i4')
        dset2 = self.f.require_dataset('foo', (10, 3), 'i2', exact=False)
        self.assertEqual(dset, dset2)
        self.assertEqual(dset2.dtype, np.dtype('i4'))
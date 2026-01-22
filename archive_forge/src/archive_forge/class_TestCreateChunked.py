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
class TestCreateChunked(BaseDataset):
    """
        Feature: Datasets can be created by manually specifying chunks
    """

    def test_create_chunks(self):
        """ Create via chunks tuple """
        dset = self.f.create_dataset('foo', shape=(100,), chunks=(10,))
        self.assertEqual(dset.chunks, (10,))

    def test_create_chunks_integer(self):
        """ Create via chunks integer """
        dset = self.f.create_dataset('foo', shape=(100,), chunks=10)
        self.assertEqual(dset.chunks, (10,))

    def test_chunks_mismatch(self):
        """ Illegal chunk size raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', shape=(100,), chunks=(200,))

    def test_chunks_false(self):
        """ Chunked format required for given storage options """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', shape=(10,), maxshape=100, chunks=False)

    def test_chunks_scalar(self):
        """ Attempting to create chunked scalar dataset raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo', shape=(), chunks=(50,))

    def test_auto_chunks(self):
        """ Auto-chunking of datasets """
        dset = self.f.create_dataset('foo', shape=(20, 100), chunks=True)
        self.assertIsInstance(dset.chunks, tuple)
        self.assertEqual(len(dset.chunks), 2)

    def test_auto_chunks_abuse(self):
        """ Auto-chunking with pathologically large element sizes """
        dset = self.f.create_dataset('foo', shape=(3,), dtype='S100000000', chunks=True)
        self.assertEqual(dset.chunks, (1,))

    def test_scalar_assignment(self):
        """ Test scalar assignment of chunked dataset """
        dset = self.f.create_dataset('foo', shape=(3, 50, 50), dtype=np.int32, chunks=(1, 50, 50))
        dset[1, :, 40] = 10
        self.assertTrue(np.all(dset[1, :, 40] == 10))
        dset[1] = 11
        self.assertTrue(np.all(dset[1] == 11))
        dset[0:2] = 12
        self.assertTrue(np.all(dset[0:2] == 12))

    def test_auto_chunks_no_shape(self):
        """ Auto-chunking of empty datasets not allowed"""
        with pytest.raises(TypeError, match='Empty') as err:
            self.f.create_dataset('foo', dtype='S100', chunks=True)
        with pytest.raises(TypeError, match='Empty') as err:
            self.f.create_dataset('foo', dtype='S100', maxshape=20)
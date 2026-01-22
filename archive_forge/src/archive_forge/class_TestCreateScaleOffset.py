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
@ut.skipIf('scaleoffset' not in h5py.filters.encode, 'SCALEOFFSET is not installed')
class TestCreateScaleOffset(BaseDataset):
    """
        Feature: Datasets can use the scale/offset filter
    """

    def test_float_fails_without_options(self):
        """ Ensure that a scale factor is required for scaleoffset compression of floating point data """
        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=float, scaleoffset=True)

    def test_non_integer(self):
        """ Check when scaleoffset is negetive"""
        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=float, scaleoffset=-0.1)

    def test_unsupport_dtype(self):
        """ Check when dtype is unsupported type"""
        with self.assertRaises(TypeError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=bool, scaleoffset=True)

    def test_float(self):
        """ Scaleoffset filter works for floating point data """
        scalefac = 4
        shape = (100, 300)
        range = 20 * 10 ** scalefac
        testdata = (np.random.rand(*shape) - 0.5) * range
        dset = self.f.create_dataset('foo', shape, dtype=float, scaleoffset=scalefac)
        assert dset.scaleoffset is not None
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        self.assertArrayEqual(readdata, testdata, precision=10 ** (-scalefac))
        assert not (readdata == testdata).all()

    def test_int(self):
        """ Scaleoffset filter works for integer data with default precision """
        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** nbits - 1, size=shape)
        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=True)
        assert dset.scaleoffset is not None
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        self.assertArrayEqual(readdata, testdata)

    def test_int_with_minbits(self):
        """ Scaleoffset filter works for integer data with specified precision """
        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** nbits, size=shape)
        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=nbits)
        self.assertTrue(dset.scaleoffset == 12)
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        self.assertArrayEqual(readdata, testdata)

    def test_int_with_minbits_lossy(self):
        """ Scaleoffset filter works for integer data with specified precision """
        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** (nbits + 1) - 1, size=shape)
        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=nbits)
        self.assertTrue(dset.scaleoffset == 12)
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        assert not (readdata == testdata).all()
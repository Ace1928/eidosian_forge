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
class TestCreateFillvalue(BaseDataset):
    """
        Feature: Datasets can be created with fill value
    """

    def test_create_fillval(self):
        """ Fill value is reflected in dataset contents """
        dset = self.f.create_dataset('foo', (10,), fillvalue=4.0)
        self.assertEqual(dset[0], 4.0)
        self.assertEqual(dset[7], 4.0)

    def test_property(self):
        """ Fill value is recoverable via property """
        dset = self.f.create_dataset('foo', (10,), fillvalue=3.0)
        self.assertEqual(dset.fillvalue, 3.0)
        self.assertNotIsInstance(dset.fillvalue, np.ndarray)

    def test_property_none(self):
        """ .fillvalue property works correctly if not set """
        dset = self.f.create_dataset('foo', (10,))
        self.assertEqual(dset.fillvalue, 0)

    def test_compound(self):
        """ Fill value works with compound types """
        dt = np.dtype([('a', 'f4'), ('b', 'i8')])
        v = np.ones((1,), dtype=dt)[0]
        dset = self.f.create_dataset('foo', (10,), dtype=dt, fillvalue=v)
        self.assertEqual(dset.fillvalue, v)
        self.assertAlmostEqual(dset[4], v)

    def test_exc(self):
        """ Bogus fill value raises ValueError """
        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (10,), dtype=[('a', 'i'), ('b', 'f')], fillvalue=42)
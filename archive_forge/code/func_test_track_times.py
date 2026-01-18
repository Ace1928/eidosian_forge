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
def test_track_times(self):
    orig = self.f.create_dataset('honda', data=np.arange(12), track_times=True)
    self.assertNotEqual(0, h5py.h5g.get_objinfo(orig._id).mtime)
    similar = self.f.create_dataset_like('hyundai', orig)
    self.assertNotEqual(0, h5py.h5g.get_objinfo(similar._id).mtime)
    orig = self.f.create_dataset('ibm', data=np.arange(12), track_times=False)
    self.assertEqual(0, h5py.h5g.get_objinfo(orig._id).mtime)
    similar = self.f.create_dataset_like('lenovo', orig)
    self.assertEqual(0, h5py.h5g.get_objinfo(similar._id).mtime)
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
def test_scalar_assignment(self):
    """ Test scalar assignment of chunked dataset """
    dset = self.f.create_dataset('foo', shape=(3, 50, 50), dtype=np.int32, chunks=(1, 50, 50))
    dset[1, :, 40] = 10
    self.assertTrue(np.all(dset[1, :, 40] == 10))
    dset[1] = 11
    self.assertTrue(np.all(dset[1] == 11))
    dset[0:2] = 12
    self.assertTrue(np.all(dset[0:2] == 12))
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
def test_string_fixed(self):
    """ Assignment of fixed-length byte string produces a fixed-length
        ascii dataset """
    self.f['x'] = np.string_('Hello there')
    ds = self.f['x']
    self.assert_string_type(ds, h5py.h5t.CSET_ASCII, variable=False)
    self.assertEqual(ds.id.get_type().get_size(), 11)
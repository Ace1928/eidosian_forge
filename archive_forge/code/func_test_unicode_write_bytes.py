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
def test_unicode_write_bytes(self):
    """ Writing valid utf-8 byte strings to a unicode vlen dataset is OK
        """
    dt = h5py.string_dtype()
    ds = self.f.create_dataset('x', (100,), dtype=dt)
    data = (u'Hello there' + chr(8244)).encode('utf8')
    ds[0] = data
    out = ds[0]
    self.assertEqual(type(out), bytes)
    self.assertEqual(out, data)
import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
class TestDtypeAssignment(BaseGroup):
    """
        Feature: Named types can be created by direct assignment of dtypes
    """

    def test_dtype(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        self.f['a'] = dtype
        self.assertIsInstance(self.f['a'], Datatype)
        self.assertEqual(self.f['a'].dtype, dtype)

    def test_name_bytes(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        self.f[b'b'] = dtype
        self.assertIsInstance(self.f[b'b'], Datatype)
import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
def test_w0(self):
    fname = self.mktemp()
    f = h5py.File(fname, 'w', rdcc_w0=0.25)
    self.assertEqual(list(f.id.get_access_plist().get_cache()), [0, 521, 1048576, 0.25])
from h5py import File
from h5py._hl.base import is_hdf5, Empty
from .common import ut, TestCase, UNICODE_FILENAMES
import numpy as np
import os
import tempfile
def test_object_parent(self):
    grp = self.f.create_group(None)
    with self.assertRaises(ValueError):
        grp.parent
    grp = self.f.create_group('bar')
    sub_grp = grp.create_group('foo')
    parent = sub_grp.parent.name
    self.assertEqual(parent, '/bar')
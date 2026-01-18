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
def test_require_shape(self):
    ds = self.f.require_dataset('foo/resizable', shape=(0, 3), maxshape=(None, 3), dtype=int)
    ds.resize(20, axis=0)
    self.f.require_dataset('foo/resizable', shape=(0, 3), maxshape=(None, 3), dtype=int)
    self.f.require_dataset('foo/resizable', shape=(20, 3), dtype=int)
    with self.assertRaises(TypeError):
        self.f.require_dataset('foo/resizable', shape=(0, 0), maxshape=(3, None), dtype=int)
    with self.assertRaises(TypeError):
        self.f.require_dataset('foo/resizable', shape=(0, 0), maxshape=(None, 5), dtype=int)
    with self.assertRaises(TypeError):
        self.f.require_dataset('foo/resizable', shape=(0, 0), maxshape=(None, 5, 2), dtype=int)
    with self.assertRaises(TypeError):
        self.f.require_dataset('foo/resizable', shape=(10, 3), dtype=int)
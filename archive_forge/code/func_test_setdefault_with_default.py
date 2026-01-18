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
def test_setdefault_with_default(self):
    """.setdefault gets default if group doesn't exist"""
    value = self.group.setdefault('e', np.array([42]))
    self.assertEqual(value, 42)
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
def test_path_type_validation(self):
    """ Access with non bytes or str types should raise an exception """
    self.f.create_group('group')
    with self.assertRaises(TypeError):
        self.f[0]
    with self.assertRaises(TypeError):
        self.f[...]
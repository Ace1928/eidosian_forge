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
def test_open_existing(self):
    """ Existing group is opened and returned """
    grp = self.f.create_group('foo')
    grp2 = self.f.require_group('foo')
    self.assertEqual(grp2, grp)
    grp3 = self.f.require_group(b'foo')
    self.assertEqual(grp3, grp)
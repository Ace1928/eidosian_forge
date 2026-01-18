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
def test_softlinks(self):
    """ Broken softlinks are contained, but their members are not """
    self.f.create_group('grp')
    self.f['/grp/soft'] = h5py.SoftLink('/mongoose')
    self.f['/grp/external'] = h5py.ExternalLink('mongoose.hdf5', '/mongoose')
    self.assertIn('/grp/soft', self.f)
    self.assertNotIn('/grp/soft/something', self.f)
    self.assertIn('/grp/external', self.f)
    self.assertNotIn('/grp/external/something', self.f)
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
def test_unicode(self):
    """ Unicode names are correctly stored """
    name = u'/Name' + chr(17664)
    group = self.f.create_group(name)
    self.assertEqual(group.name, name)
    self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_UTF8)
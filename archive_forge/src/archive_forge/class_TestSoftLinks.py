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
class TestSoftLinks(BaseGroup):
    """
        Feature: Create and manage soft links with the high-level interface
    """

    def test_spath(self):
        """ SoftLink path attribute """
        sl = SoftLink('/foo')
        self.assertEqual(sl.path, '/foo')

    def test_srepr(self):
        """ SoftLink path repr """
        sl = SoftLink('/foo')
        self.assertIsInstance(repr(sl), str)

    def test_create(self):
        """ Create new soft link by assignment """
        g = self.f.create_group('new')
        sl = SoftLink('/new')
        self.f['alias'] = sl
        g2 = self.f['alias']
        self.assertEqual(g, g2)

    def test_exc(self):
        """ Opening dangling soft link results in KeyError """
        self.f['alias'] = SoftLink('new')
        with self.assertRaises(KeyError):
            self.f['alias']
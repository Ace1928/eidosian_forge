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
class TestRequire(BaseGroup):
    """
        Feature: Groups can be auto-created, or opened via .require_group
    """

    def test_open_existing(self):
        """ Existing group is opened and returned """
        grp = self.f.create_group('foo')
        grp2 = self.f.require_group('foo')
        self.assertEqual(grp2, grp)
        grp3 = self.f.require_group(b'foo')
        self.assertEqual(grp3, grp)

    def test_create(self):
        """ Group is created if it doesn't exist """
        grp = self.f.require_group('foo')
        self.assertIsInstance(grp, Group)
        self.assertEqual(grp.name, '/foo')

    def test_require_exception(self):
        """ Opening conflicting object results in TypeError """
        self.f.create_dataset('foo', (1,), 'f')
        with self.assertRaises(TypeError):
            self.f.require_group('foo')

    def test_intermediate_create_dataset(self):
        """ Intermediate is created if it doesn't exist """
        dt = h5py.string_dtype()
        self.f.require_dataset('foo/bar/baz', (1,), dtype=dt)
        group = self.f.get('foo')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar')
        assert isinstance(group, Group)

    def test_intermediate_create_group(self):
        dt = h5py.string_dtype()
        self.f.require_group('foo/bar/baz')
        group = self.f.get('foo')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar/baz')
        assert isinstance(group, Group)

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
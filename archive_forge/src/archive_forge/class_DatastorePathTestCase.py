from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
class DatastorePathTestCase(base.TestCase):
    """Test the DatastorePath object."""

    def test_ds_path(self):
        p = datastore.DatastorePath('dsname', 'a/b/c', 'file.iso')
        self.assertEqual('[dsname] a/b/c/file.iso', str(p))
        self.assertEqual('a/b/c/file.iso', p.rel_path)
        self.assertEqual('a/b/c', p.parent.rel_path)
        self.assertEqual('[dsname] a/b/c', str(p.parent))
        self.assertEqual('dsname', p.datastore)
        self.assertEqual('file.iso', p.basename)
        self.assertEqual('a/b/c', p.dirname)

    def test_ds_path_no_ds_name(self):
        bad_args = [('', ['a/b/c', 'file.iso']), (None, ['a/b/c', 'file.iso'])]
        for t in bad_args:
            self.assertRaises(ValueError, datastore.DatastorePath, t[0], *t[1])

    def test_ds_path_invalid_path_components(self):
        bad_args = [('dsname', [None]), ('dsname', ['', None]), ('dsname', ['a', None]), ('dsname', ['a', None, 'b']), ('dsname', [None, '']), ('dsname', [None, 'b'])]
        for t in bad_args:
            self.assertRaises(ValueError, datastore.DatastorePath, t[0], *t[1])

    def test_ds_path_no_subdir(self):
        args = [('dsname', ['', 'x.vmdk']), ('dsname', ['x.vmdk'])]
        canonical_p = datastore.DatastorePath('dsname', 'x.vmdk')
        self.assertEqual('[dsname] x.vmdk', str(canonical_p))
        self.assertEqual('', canonical_p.dirname)
        self.assertEqual('x.vmdk', canonical_p.basename)
        self.assertEqual('x.vmdk', canonical_p.rel_path)
        for t in args:
            p = datastore.DatastorePath(t[0], *t[1])
            self.assertEqual(str(canonical_p), str(p))

    def test_ds_path_ds_only(self):
        args = [('dsname', []), ('dsname', ['']), ('dsname', ['', ''])]
        canonical_p = datastore.DatastorePath('dsname')
        self.assertEqual('[dsname]', str(canonical_p))
        self.assertEqual('', canonical_p.rel_path)
        self.assertEqual('', canonical_p.basename)
        self.assertEqual('', canonical_p.dirname)
        for t in args:
            p = datastore.DatastorePath(t[0], *t[1])
            self.assertEqual(str(canonical_p), str(p))
            self.assertEqual(canonical_p.rel_path, p.rel_path)

    def test_ds_path_equivalence(self):
        args = [('dsname', ['a/b/c/', 'x.vmdk']), ('dsname', ['a/', 'b/c/', 'x.vmdk']), ('dsname', ['a', 'b', 'c', 'x.vmdk']), ('dsname', ['a/b/c', 'x.vmdk'])]
        canonical_p = datastore.DatastorePath('dsname', 'a/b/c', 'x.vmdk')
        for t in args:
            p = datastore.DatastorePath(t[0], *t[1])
            self.assertEqual(str(canonical_p), str(p))
            self.assertEqual(canonical_p.datastore, p.datastore)
            self.assertEqual(canonical_p.rel_path, p.rel_path)
            self.assertEqual(str(canonical_p.parent), str(p.parent))

    def test_ds_path_non_equivalence(self):
        args = [('dsname', ['/a', 'b', 'c', 'x.vmdk']), ('dsname', ['/a/b/c/', 'x.vmdk']), ('dsname', ['a/b/c', '/x.vmdk']), ('dsname', ['a/b/c/', ' x.vmdk']), ('dsname', ['a/', ' b/c/', 'x.vmdk']), ('dsname', [' a', 'b', 'c', 'x.vmdk']), ('dsname', ['/a/b/c/', 'x.vmdk ']), ('dsname', ['a/b/c/ ', 'x.vmdk'])]
        canonical_p = datastore.DatastorePath('dsname', 'a/b/c', 'x.vmdk')
        for t in args:
            p = datastore.DatastorePath(t[0], *t[1])
            self.assertNotEqual(str(canonical_p), str(p))

    def test_equal(self):
        a = datastore.DatastorePath('ds_name', 'a')
        b = datastore.DatastorePath('ds_name', 'a')
        self.assertEqual(a, b)

    def test_join(self):
        p = datastore.DatastorePath('ds_name', 'a')
        ds_path = p.join('b')
        self.assertEqual('[ds_name] a/b', str(ds_path))
        p = datastore.DatastorePath('ds_name', 'a')
        ds_path = p.join()
        bad_args = [[None], ['', None], ['a', None], ['a', None, 'b']]
        for arg in bad_args:
            self.assertRaises(ValueError, p.join, *arg)

    def test_ds_path_parse(self):
        p = datastore.DatastorePath.parse('[dsname]')
        self.assertEqual('dsname', p.datastore)
        self.assertEqual('', p.rel_path)
        p = datastore.DatastorePath.parse('[dsname] folder')
        self.assertEqual('dsname', p.datastore)
        self.assertEqual('folder', p.rel_path)
        p = datastore.DatastorePath.parse('[dsname] folder/file')
        self.assertEqual('dsname', p.datastore)
        self.assertEqual('folder/file', p.rel_path)
        for p in [None, '']:
            self.assertRaises(ValueError, datastore.DatastorePath.parse, p)
        for p in ['bad path', '/a/b/c', 'a/b/c']:
            self.assertRaises(IndexError, datastore.DatastorePath.parse, p)
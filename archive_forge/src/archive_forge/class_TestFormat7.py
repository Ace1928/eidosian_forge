import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
class TestFormat7(TestCaseWithTransport):

    def test_attribute__fetch_order(self):
        """Weaves need topological data insertion."""
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control)
        self.assertEqual('topological', repo._format._fetch_order)

    def test_attribute__fetch_uses_deltas(self):
        """Weaves do not reuse deltas."""
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control)
        self.assertEqual(False, repo._format._fetch_uses_deltas)

    def test_attribute__fetch_reconcile(self):
        """Weave repositories need a reconcile after fetch."""
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control)
        self.assertEqual(True, repo._format._fetch_reconcile)

    def test_disk_layout(self):
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control)
        repo.lock_write()
        repo.unlock()
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Repository format 7', f.read())
        self.assertTrue(S_ISDIR(t.stat('revision-store').st_mode))
        self.assertTrue(S_ISDIR(t.stat('weaves').st_mode))
        with t.get('inventory.weave') as f:
            self.assertEqualDiff(b'# bzr weave file v5\nw\nW\n', f.read())
        control.create_branch()
        tree = control.create_workingtree()
        tree.add(['foo'], ['file'], ids=[b'Foo:Bar'])
        tree.put_file_bytes_non_atomic('foo', b'content\n')
        try:
            tree.commit('first post', rev_id=b'first')
        except IllegalPath:
            if sys.platform != 'win32':
                raise
            self.knownFailure('Foo:Bar cannot be used as a file-id on windows in repo format 7')
            return
        with t.get('weaves/74/Foo%3ABar.weave') as f:
            self.assertEqualDiff(b'# bzr weave file v5\ni\n1 7fe70820e08a1aac0ef224d9c66ab66831cc4ab1\nn first\n\nw\n{ 0\n. content\n}\nW\n', f.read())

    def test_shared_disk_layout(self):
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control, shared=True)
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Repository format 7', f.read())
        with t.get('shared-storage') as f:
            self.assertEqualDiff(b'', f.read())
        self.assertTrue(S_ISDIR(t.stat('revision-store').st_mode))
        self.assertTrue(S_ISDIR(t.stat('weaves').st_mode))
        with t.get('inventory.weave') as f:
            self.assertEqualDiff(b'# bzr weave file v5\nw\nW\n', f.read())
        self.assertFalse(t.has('branch-lock'))

    def test_creates_lockdir(self):
        """Make sure it appears to be controlled by a LockDir existence"""
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control, shared=True)
        t = control.get_repository_transport(None)
        self.assertFalse(t.has('lock/held/info'))
        with repo.lock_write():
            self.assertTrue(t.has('lock/held/info'))

    def test_uses_lockdir(self):
        """repo format 7 actually locks on lockdir"""
        base_url = self.get_url()
        control = BzrDirMetaFormat1().initialize(base_url)
        repo = RepositoryFormat7().initialize(control, shared=True)
        t = control.get_repository_transport(None)
        repo.lock_write()
        repo.unlock()
        del repo
        repo = Repository.open(base_url)
        repo.lock_write()
        self.assertTrue(t.has('lock/held/info'))
        repo.unlock()
        self.assertFalse(t.has('lock/held/info'))

    def test_shared_no_tree_disk_layout(self):
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control, shared=True)
        repo.set_make_working_trees(False)
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Repository format 7', f.read())
        with t.get('shared-storage') as f:
            self.assertEqualDiff(b'', f.read())
        with t.get('no-working-trees') as f:
            self.assertEqualDiff(b'', f.read())
        repo.set_make_working_trees(True)
        self.assertFalse(t.has('no-working-trees'))
        self.assertTrue(S_ISDIR(t.stat('revision-store').st_mode))
        self.assertTrue(S_ISDIR(t.stat('weaves').st_mode))
        with t.get('inventory.weave') as f:
            self.assertEqualDiff(b'# bzr weave file v5\nw\nW\n', f.read())

    def test_supports_external_lookups(self):
        control = BzrDirMetaFormat1().initialize(self.get_url())
        repo = RepositoryFormat7().initialize(control)
        self.assertFalse(repo._format.supports_external_lookups)
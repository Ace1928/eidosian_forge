from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestFormatKnit1(TestCaseWithTransport):

    def test_attribute__fetch_order(self):
        """Knits need topological data insertion."""
        repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
        self.assertEqual('topological', repo._format._fetch_order)

    def test_attribute__fetch_uses_deltas(self):
        """Knits reuse deltas."""
        repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
        self.assertEqual(True, repo._format._fetch_uses_deltas)

    def test_disk_layout(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        repo = knitrepo.RepositoryFormatKnit1().initialize(control)
        repo.lock_write()
        repo.unlock()
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Knit Repository Format 1', f.read())
        self.assertTrue(S_ISDIR(t.stat('knits').st_mode))
        self.check_knits(t)
        control.create_branch()
        tree = control.create_workingtree()
        tree.add(['foo'], ['file'], [b'Nasty-IdC:'])
        tree.put_file_bytes_non_atomic('foo', b'')
        tree.commit('1st post', rev_id=b'foo')
        self.assertHasKnit(t, 'knits/e8/%254easty-%2549d%2543%253a', b'\nfoo fulltext 0 81  :')

    def assertHasKnit(self, t, knit_name, extra_content=b''):
        """Assert that knit_name exists on t."""
        with t.get(knit_name + '.kndx') as f:
            self.assertEqualDiff(b'# bzr knit index 8\n' + extra_content, f.read())

    def check_knits(self, t):
        """check knit content for a repository."""
        self.assertHasKnit(t, 'inventory')
        self.assertHasKnit(t, 'revisions')
        self.assertHasKnit(t, 'signatures')

    def test_shared_disk_layout(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        knitrepo.RepositoryFormatKnit1().initialize(control, shared=True)
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Knit Repository Format 1', f.read())
        with t.get('shared-storage') as f:
            self.assertEqualDiff(b'', f.read())
        self.assertTrue(S_ISDIR(t.stat('knits').st_mode))
        self.check_knits(t)

    def test_shared_no_tree_disk_layout(self):
        control = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        repo = knitrepo.RepositoryFormatKnit1().initialize(control, shared=True)
        repo.set_make_working_trees(False)
        t = control.get_repository_transport(None)
        with t.get('format') as f:
            self.assertEqualDiff(b'Bazaar-NG Knit Repository Format 1', f.read())
        with t.get('shared-storage') as f:
            self.assertEqualDiff(b'', f.read())
        with t.get('no-working-trees') as f:
            self.assertEqualDiff(b'', f.read())
        repo.set_make_working_trees(True)
        self.assertFalse(t.has('no-working-trees'))
        self.assertTrue(S_ISDIR(t.stat('knits').st_mode))
        self.check_knits(t)

    def test_deserialise_sets_root_revision(self):
        """We must have a inventory.root.revision

        Old versions of the XML5 serializer did not set the revision_id for
        the whole inventory. So we grab the one from the expected text. Which
        is valid when the api is not being abused.
        """
        repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
        inv_xml = b'<inventory format="5">\n</inventory>\n'
        inv = repo._deserialise_inventory(b'test-rev-id', [inv_xml])
        self.assertEqual(b'test-rev-id', inv.root.revision)

    def test_deserialise_uses_global_revision_id(self):
        """If it is set, then we re-use the global revision id"""
        repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
        inv_xml = b'<inventory format="5" revision_id="other-rev-id">\n</inventory>\n'
        self.assertRaises(AssertionError, repo._deserialise_inventory, b'test-rev-id', [inv_xml])
        inv = repo._deserialise_inventory(b'other-rev-id', [inv_xml])
        self.assertEqual(b'other-rev-id', inv.root.revision)

    def test_supports_external_lookups(self):
        repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
        self.assertFalse(repo._format.supports_external_lookups)
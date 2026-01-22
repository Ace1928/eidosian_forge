import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
class TestReconcileWithIncorrectRevisionCache(TestReconcile):
    """Ancestry data gets cached in knits and weaves should be reconcilable.

    This class tests that reconcile can correct invalid caches (such as after
    a reconcile).
    """

    def setUp(self):
        self.reduceLockdirTimeout()
        super().setUp()
        t = self.get_transport()
        self.first_tree = self.make_branch_and_tree('wrong-first-parent')
        self.second_tree = self.make_branch_and_tree('reversed-secondary-parents')
        for t in [self.first_tree, self.second_tree]:
            t.commit('1', rev_id=b'1')
            uncommit(t.branch, tree=t)
            t.commit('2', rev_id=b'2')
            uncommit(t.branch, tree=t)
            t.commit('3', rev_id=b'3')
            uncommit(t.branch, tree=t)
        repo_secondary = self.second_tree.branch.repository
        repo = self.first_tree.branch.repository
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id=b'wrong-first-parent')
        inv.root.revision = b'wrong-first-parent'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, b'wrong-first-parent'), [], [])
        sha1 = repo.add_inventory(b'wrong-first-parent', inv, [b'2', b'1'])
        rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=b'wrong-first-parent')
        rev.parent_ids = [b'1', b'2']
        repo.add_revision(b'wrong-first-parent', rev)
        repo.commit_write_group()
        repo.unlock()
        repo = repo_secondary
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id=b'wrong-secondary-parent')
        inv.root.revision = b'wrong-secondary-parent'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, b'wrong-secondary-parent'), [], [])
        sha1 = repo.add_inventory(b'wrong-secondary-parent', inv, [b'1', b'3', b'2'])
        rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=b'wrong-secondary-parent')
        rev.parent_ids = [b'1', b'2', b'3']
        repo.add_revision(b'wrong-secondary-parent', rev)
        repo.commit_write_group()
        repo.unlock()

    def test_reconcile_wrong_order(self):
        repo = self.first_tree.branch.repository
        with repo.lock_read():
            g = repo.get_graph()
            if g.get_parent_map([b'wrong-first-parent'])[b'wrong-first-parent'] == (b'1', b'2'):
                raise TestSkipped('wrong-first-parent is not setup for testing')
        self.checkUnreconciled(repo.controldir, repo.reconcile())
        reconciler = repo.reconcile(thorough=True)
        self.assertEqual(1, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        g = repo.get_graph()
        self.assertEqual({b'wrong-first-parent': (b'1', b'2')}, g.get_parent_map([b'wrong-first-parent']))

    def test_reconcile_wrong_order_secondary_inventory(self):
        repo = self.second_tree.branch.repository
        self.checkUnreconciled(repo.controldir, repo.reconcile())
        self.checkUnreconciled(repo.controldir, repo.reconcile(thorough=True))
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
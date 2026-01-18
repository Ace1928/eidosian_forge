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
def test_reweave_inventory_preserves_a_revision_with_ghosts(self):
    d = BzrDir.open(self.get_url('inventory_one_ghost'))
    reconciler = d.open_repository().reconcile(thorough=True)
    self.assertEqual(0, reconciler.inconsistent_parents)
    self.assertEqual(0, reconciler.garbage_inventories)
    repo = d.open_repository()
    repo.get_inventory(b'ghost')
    self.assertThat([b'ghost', b'the_ghost'], MatchesAncestry(repo, b'ghost'))
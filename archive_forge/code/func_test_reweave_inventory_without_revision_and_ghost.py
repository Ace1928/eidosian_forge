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
def test_reweave_inventory_without_revision_and_ghost(self):
    d_url = self.get_url('inventory_without_revision_and_ghost')
    d = BzrDir.open(d_url)
    repo = d.open_repository()
    if not repo._reconcile_does_inventory_gc:
        raise TestSkipped('Irrelevant test')
    self.check_thorough_reweave_missing_revision(d, repo.reconcile, thorough=True)
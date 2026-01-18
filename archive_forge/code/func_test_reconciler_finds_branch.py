from breezy import errors, tests
from breezy.bzr import bzrdir
from breezy.reconcile import Reconciler, reconcile
from breezy.tests import per_repository
def test_reconciler_finds_branch(self):
    a_branch = self.make_branch('a_branch')
    reconciler = Reconciler(a_branch.controldir)
    result = reconciler.reconcile()
    self.assertEqual(0, result.inconsistent_parents)
    self.assertEqual(0, result.garbage_inventories)
    self.assertIs(False, result.fixed_branch_history)
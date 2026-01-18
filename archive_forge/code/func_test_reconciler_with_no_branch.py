from breezy import errors, tests
from breezy.bzr import bzrdir
from breezy.reconcile import Reconciler, reconcile
from breezy.tests import per_repository
def test_reconciler_with_no_branch(self):
    repo = self.make_repository('repo')
    reconciler = Reconciler(repo.controldir)
    result = reconciler.reconcile()
    self.assertEqual(0, result.inconsistent_parents)
    self.assertEqual(0, result.garbage_inventories)
    self.assertIs(None, result.fixed_branch_history)
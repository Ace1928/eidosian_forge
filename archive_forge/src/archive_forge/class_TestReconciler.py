from breezy import errors, tests
from breezy.bzr import bzrdir
from breezy.reconcile import Reconciler, reconcile
from breezy.tests import per_repository
class TestReconciler(tests.TestCaseWithTransport):

    def test_reconciler_with_no_branch(self):
        repo = self.make_repository('repo')
        reconciler = Reconciler(repo.controldir)
        result = reconciler.reconcile()
        self.assertEqual(0, result.inconsistent_parents)
        self.assertEqual(0, result.garbage_inventories)
        self.assertIs(None, result.fixed_branch_history)

    def test_reconciler_finds_branch(self):
        a_branch = self.make_branch('a_branch')
        reconciler = Reconciler(a_branch.controldir)
        result = reconciler.reconcile()
        self.assertEqual(0, result.inconsistent_parents)
        self.assertEqual(0, result.garbage_inventories)
        self.assertIs(False, result.fixed_branch_history)
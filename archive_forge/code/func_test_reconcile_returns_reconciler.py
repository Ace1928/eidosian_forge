from breezy import errors, reconcile
from breezy.bzr.branch import BzrBranch
from breezy.symbol_versioning import deprecated_in
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_reconcile_returns_reconciler(self):
    a_branch = self.make_branch('a_branch')
    result = a_branch.reconcile()
    self.assertIsInstance(result, reconcile.ReconcileResult)
    self.assertIs(False, getattr(result, 'fixed_history', False))
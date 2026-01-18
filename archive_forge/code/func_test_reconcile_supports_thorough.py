from breezy import errors, reconcile
from breezy.bzr.branch import BzrBranch
from breezy.symbol_versioning import deprecated_in
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_reconcile_supports_thorough(self):
    a_branch = self.make_branch('a_branch')
    a_branch.reconcile(thorough=False)
    a_branch.reconcile(thorough=True)
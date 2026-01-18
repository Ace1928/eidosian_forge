from breezy import errors, reconcile
from breezy.bzr.branch import BzrBranch
from breezy.symbol_versioning import deprecated_in
from breezy.tests import TestNotApplicable
from breezy.tests.per_branch import TestCaseWithBranch
def test_reconcile_fixes_invalid_revhistory(self):
    if not isinstance(self.branch_format, BzrBranch):
        raise TestNotApplicable('test only applies to bzr formats')
    tree = self.make_branch_and_tree('test')
    r1 = tree.commit('one')
    r2 = tree.commit('two')
    r3 = tree.commit('three')
    r4 = tree.commit('four')
    tree.set_parent_ids([r1])
    tree.branch.set_last_revision_info(1, r1)
    r2b = tree.commit('two-b')
    tree.set_parent_ids([r4, r2b])
    tree.branch.set_last_revision_info(4, r4)
    r5 = tree.commit('five')
    try:
        self.applyDeprecated(deprecated_in((2, 4, 0)), tree.branch.set_revision_history, [r1, r2b, r5])
        if tree.branch.last_revision_info() != (3, r5):
            tree.branch.set_last_revision_info(3, r5)
    except errors.NotLefthandHistory:
        tree.branch.set_last_revision_info(3, r5)
    self.assertEqual((3, r5), tree.branch.last_revision_info())
    reconciler = tree.branch.reconcile()
    self.assertEqual((5, r5), tree.branch.last_revision_info())
    self.assertIs(True, reconciler.fixed_history)
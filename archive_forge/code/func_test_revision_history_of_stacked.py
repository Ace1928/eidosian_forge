from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_revision_history_of_stacked(self):
    stack_on = self.make_branch_and_tree('stack-on')
    rev1 = stack_on.commit('first commit')
    try:
        stacked_dir = stack_on.controldir.sprout(self.get_url('stacked'), stacked=True)
    except unstackable_format_errors as e:
        raise TestNotApplicable('Format does not support stacking.')
    try:
        stacked = stacked_dir.open_workingtree()
    except errors.NoWorkingTree:
        stacked = stacked_dir.open_branch().create_checkout('stacked-checkout', lightweight=True)
    tree = stacked.branch.create_checkout('local')
    rev2 = tree.commit('second commit')
    repo = stacked.branch.repository.controldir.open_repository()
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual({}, repo.get_parent_map([rev1]))
    self.assertEqual((2, rev2), stacked.branch.last_revision_info())
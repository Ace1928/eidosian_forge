from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_to_colocated_existing_inactive(self):
    tree, rev_1 = self.create_simple_tree()
    target_tree = self.make_branch_and_tree('dir')
    rev_o = target_tree.commit('another')
    try:
        target_tree.branch.controldir.create_branch(name='colo')
    except NoColocatedBranchSupport:
        raise TestNotApplicable('no colocated branch support')
    try:
        result = target_tree.branch.controldir.push_branch(tree.branch, name='colo')
    except NoColocatedBranchSupport:
        raise TestNotApplicable('no colocated branch support')
    target_branch = target_tree.branch.controldir.open_branch(name='colo')
    self.assertEqual(tree.branch, result.source_branch)
    self.assertEqual(target_tree.last_revision(), rev_o)
    self.assertEqual(target_tree.branch.last_revision(), rev_o)
    self.assertEqual(target_branch.base, result.target_branch.base)
    self.assertEqual(target_branch.base, tree.branch.get_push_location())
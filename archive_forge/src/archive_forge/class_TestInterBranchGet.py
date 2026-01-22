from breezy import branch
from breezy.tests.per_interbranch import TestCaseWithInterBranch
class TestInterBranchGet(TestCaseWithInterBranch):

    def test_gets_right_inter(self):
        self.tree1 = self.make_from_branch_and_tree('tree1')
        branch2 = self.make_to_branch('tree2')
        self.assertIs(branch.InterBranch.get(self.tree1.branch, branch2).__class__, self.interbranch_class)
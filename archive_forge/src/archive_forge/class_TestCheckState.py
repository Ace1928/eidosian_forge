from breezy import errors, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestCheckState(TestCaseWithState):

    def test_check_state(self):
        tree = self.make_branch_and_tree('tree')
        tree.check_state()

    def test_check_broken_dirstate(self):
        tree = self.make_tree_with_broken_dirstate('tree')
        self.assertRaises(errors.BzrError, tree.check_state)
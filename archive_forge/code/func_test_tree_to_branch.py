from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_tree_to_branch(self):
    tree = self.make_branch_and_tree('tree')
    self.run_bzr('reconfigure --branch tree')
    self.assertRaises(errors.NoWorkingTree, workingtree.WorkingTree.open, 'tree')
from breezy import workingtree
from breezy.tests import TestCaseWithTransport
def test_repair_corrupted_dirstate(self):
    tree = self.make_initial_tree()
    self.break_dirstate(tree)
    self.run_bzr('repair-workingtree -d tree')
    tree = workingtree.WorkingTree.open('tree')
    tree.check_state()
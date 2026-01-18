from breezy import uncommit
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_uncommit_to_null(self):
    tree = self.make_branch_and_tree('branch')
    tree.lock_write()
    revid = tree.commit('a revision')
    tree.unlock()
    uncommit.uncommit(tree.branch, tree=tree)
    self.assertEqual([], tree.get_parent_ids())
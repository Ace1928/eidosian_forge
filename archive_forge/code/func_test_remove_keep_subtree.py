from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_keep_subtree(self):
    """Check that a directory is unversioned but not deleted."""
    tree = self.make_branch_and_tree('.')
    subtree = self.make_branch_and_tree('subtree')
    subtree.commit('')
    tree.add('subtree')
    tree.remove('subtree')
    self.assertRemovedAndNotDeleted('subtree')
from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_keep(self):
    """Check that files and directories are unversioned but not deleted."""
    tree = self.get_tree(TestRemove.files)
    tree.add(TestRemove.files)
    tree.remove(TestRemove.files)
    self.assertRemovedAndNotDeleted(TestRemove.files)
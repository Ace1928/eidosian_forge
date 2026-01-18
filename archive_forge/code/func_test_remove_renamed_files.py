from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_renamed_files(self):
    """Check that files are removed even if they are renamed."""
    tree = self.get_committed_tree(TestRemove.files)
    for f in TestRemove.rfiles:
        tree.rename_one(f, f + 'x')
    rfilesx = ['bx/cx', 'bx', 'ax', 'dx']
    self.assertPathExists(rfilesx)
    tree.remove(rfilesx, keep_files=False)
    self.assertRemovedAndDeleted(rfilesx)
    tree._validate()
from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_renamed_changed_files(self):
    """Check that files that are renamed and changed are backed up."""
    tree = self.get_committed_tree(TestRemove.files)
    for f in TestRemove.rfiles:
        tree.rename_one(f, f + 'x')
    rfilesx = ['bx/cx', 'bx', 'ax', 'dx']
    self.build_tree_contents([('ax', b'changed and renamed!'), ('bx/cx', b'changed and renamed!')])
    self.assertPathExists(rfilesx)
    tree.remove(rfilesx, keep_files=False)
    self.assertNotInWorkingTree(rfilesx)
    self.assertPathExists(['bx.~1~/cx.~1~', 'bx.~1~', 'ax.~1~'])
    if tree.supports_rename_tracking() or not tree.has_versioned_directories():
        self.assertPathDoesNotExist('dx.~1~')
    else:
        self.assertPathExists('dx.~1~')
    tree._validate()
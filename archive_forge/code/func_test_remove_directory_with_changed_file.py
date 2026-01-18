from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_directory_with_changed_file(self):
    """Backup directories with changed files."""
    files = ['b/', 'b/c']
    tree = self.get_committed_tree(files)
    self.build_tree_contents([('b/c', b'some other new content!')])
    tree.remove('b', keep_files=False)
    self.assertPathExists('b.~1~/c.~1~')
    self.assertNotInWorkingTree(files)
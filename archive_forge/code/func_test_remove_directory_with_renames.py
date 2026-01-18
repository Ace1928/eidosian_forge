from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_directory_with_renames(self):
    """Delete directory with renames in or out."""
    files = ['a/', 'a/file', 'a/directory/', 'a/directory/stuff', 'b/']
    files_to_move = ['a/file', 'a/directory/']
    tree = self.get_committed_tree(files)
    tree.move(['a/file', 'a/directory'], to_dir='b')
    moved_files = ['b/file', 'b/directory/']
    self.assertRemovedAndDeleted(files_to_move)
    self.assertInWorkingTree(moved_files)
    self.assertPathExists(moved_files)
    tree.remove('a', keep_files=False)
    self.assertRemovedAndDeleted(['a/'])
    tree.remove('b', keep_files=False)
    self.assertRemovedAndDeleted(['b/'])
    tree._validate()
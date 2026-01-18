from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_uncommitted_removed_file(self):
    tree = self.get_committed_tree(['a'])
    tree.remove('a', keep_files=False)
    tree.remove('a', keep_files=False)
    self.assertPathDoesNotExist('a')
    tree._validate()
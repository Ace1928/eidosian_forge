from breezy import ignores, osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_remove_unknown_ignored_files(self):
    """Unknown ignored files should be deleted."""
    tree = self.get_committed_tree(['b/'])
    ignores.add_runtime_ignores(['*ignored*'])
    self.build_tree(['unknown_ignored_file'])
    self.assertNotEqual(None, tree.is_ignored('unknown_ignored_file'))
    tree.remove('unknown_ignored_file', keep_files=False)
    self.assertRemovedAndDeleted('unknown_ignored_file')
    self.build_tree(['b/unknown_ignored_file', 'b/unknown_ignored_dir/'])
    self.assertNotEqual(None, tree.is_ignored('b/unknown_ignored_file'))
    self.assertNotEqual(None, tree.is_ignored('b/unknown_ignored_dir'))
    tree.remove('b', keep_files=False)
    self.assertRemovedAndDeleted('b')
    tree._validate()
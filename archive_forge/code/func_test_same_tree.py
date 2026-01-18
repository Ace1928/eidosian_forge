from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_same_tree(self):
    """Test basis_tree when working tree hasn't been modified."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file', 'dir/', 'dir/subfile'])
    tree.add(['file', 'dir', 'dir/subfile'])
    revision_id = tree.commit('initial tree')
    basis_tree = tree.basis_tree()
    with basis_tree.lock_read():
        self.assertEqual(revision_id, basis_tree.get_revision_id())
        self.assertEqual(['', 'dir', 'dir/subfile', 'file'], sorted((info[0] for info in basis_tree.list_files(True))))
import os
from breezy import merge, tests, transform, workingtree
def test_revert_file_in_deleted_dir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'dir/file1', 'dir/file2'])
    tree.add(['dir', 'dir/file1', 'dir/file2'], ids=[b'dir-id', b'file1-id', b'file2-id'])
    tree.commit('Added files')
    os.unlink('dir/file1')
    os.unlink('dir/file2')
    os.rmdir('dir')
    tree.remove(['dir/', 'dir/file1', 'dir/file2'])
    tree.revert(['dir/file1'])
    self.assertPathExists('dir/file1')
    self.assertPathDoesNotExist('dir/file2')
    self.assertEqual(b'dir-id', tree.path2id('dir'))
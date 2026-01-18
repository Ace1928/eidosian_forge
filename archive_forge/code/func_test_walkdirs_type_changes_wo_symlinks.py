import os
from breezy.tests.features import SymlinkFeature
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_walkdirs_type_changes_wo_symlinks(self):
    tree = self.make_branch_and_tree('.')
    paths = ['file1', 'dir1/']
    self.build_tree(paths)
    tree.add(paths)
    tree.controldir.root_transport.delete_tree('dir1')
    tree.controldir.root_transport.delete('file1')
    changed_paths = ['dir1', 'file1/']
    self.build_tree(changed_paths)
    dir1_stat = os.lstat('dir1')
    file1_stat = os.lstat('file1')
    if tree.has_versioned_directories():
        expected_dirblocks = [('', [('dir1', 'dir1', 'file', dir1_stat, 'directory'), ('file1', 'file1', 'directory', file1_stat, 'file')]), ('dir1', []), ('file1', [])]
    else:
        expected_dirblocks = [('', [('dir1', 'dir1', 'file', dir1_stat, None), ('file1', 'file1', 'directory', file1_stat, 'file')]), ('file1', [])]
    with tree.lock_read():
        result = list(tree.walkdirs())
    for pos, item in enumerate(expected_dirblocks):
        self.assertEqual(item, result[pos])
    self.assertEqual(len(expected_dirblocks), len(result))
import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_symlink_changes_to_dir(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a@', 'target')])
    tree.smart_add(['tree/a'])
    tree.commit('add symlink')
    os.unlink('tree/a')
    self.build_tree_contents([('tree/a/',), ('tree/a/f', b'content')])
    tree.smart_add(['tree/a/f'])
    tree.commit('change to dir')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual([], list(tree.iter_changes(tree.basis_tree())))
    self.assertEqual(['a', 'a/f'], sorted((info[0] for info in tree.list_files())))
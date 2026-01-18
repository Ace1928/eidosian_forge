import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_only_in_target_missing_subtree_specific_bug_367632(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree2/a-dir/', 'tree2/a-dir/a-file'])
    tree2.add(['a-dir', 'a-dir/a-file'], ids=[b'dir-id', b'file-id'])
    os.unlink('tree2/a-dir/a-file')
    os.rmdir('tree2/a-dir')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_missing_in('a-dir', tree2)
    root_id = tree1.path2id('')
    expected = [InventoryTreeChange(b'dir-id', (None, 'a-dir'), False, (False, True), (None, root_id), (None, 'a-dir'), (None, None), (None, False)), InventoryTreeChange(b'file-id', (None, 'a-dir/a-file'), False, (False, True), (None, b'dir-id'), (None, 'a-file'), (None, None), (None, False))]
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['']))
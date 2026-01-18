import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_only_in_target_and_missing(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    self.build_tree(['tree2/file'])
    tree2.add(['file'], ids=[b'file-id'])
    os.unlink('tree2/file')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_missing_in('file', tree2)
    root_id = tree1.path2id('')
    expected = [InventoryTreeChange(b'file-id', (None, 'file'), False, (False, True), (None, root_id), (None, 'file'), (None, None), (None, False))]
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
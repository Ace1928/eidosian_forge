import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_unchanged_with_renames_and_modifications(self):
    """want_unchanged should generate a list of unchanged entries."""
    tree1 = self.make_branch_and_tree('1')
    tree2 = self.make_to_branch_and_tree('2')
    tree1 = self.get_tree_no_parents_abc_content(tree1)
    tree2 = self.get_tree_no_parents_abc_content_5(tree2)
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual(sorted([self.unchanged(tree1, ''), self.unchanged(tree1, 'b'), InventoryTreeChange(b'a-id', ('a', 'd'), True, (True, True), (b'root-id', b'root-id'), ('a', 'd'), ('file', 'file'), (False, False)), self.unchanged(tree1, 'b/c')]), self.do_iter_changes(tree1, tree2, include_unchanged=True))
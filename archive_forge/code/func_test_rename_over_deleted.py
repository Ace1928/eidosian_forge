import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_rename_over_deleted(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree1/c', b'c contents\n'), ('tree1/d', b'd contents\n'), ('tree2/a', b'b contents\n'), ('tree2/d', b'c contents\n')])
    tree1.add(['a', 'b', 'c', 'd'], ids=[b'a-id', b'b-id', b'c-id', b'd-id'])
    tree2.add(['a', 'd'], ids=[b'b-id', b'c-id'])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected = self.sorted([self.deleted(tree1, 'a'), self.deleted(tree1, 'd'), self.renamed(tree1, tree2, 'b', 'a', False), self.renamed(tree1, tree2, 'c', 'd', False)])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2))
    self.check_has_changes(True, tree1, tree2)
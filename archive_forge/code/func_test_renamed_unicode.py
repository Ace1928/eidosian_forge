import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_renamed_unicode(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    a_id = 'α-id'.encode()
    rename_id = 'ω_rename_id'.encode()
    try:
        self.build_tree(['tree1/α/', 'tree2/α/'])
    except UnicodeError:
        raise tests.TestSkipped('Could not create Unicode files.')
    self.build_tree_contents([('tree1/ω-source', b'contents\n'), ('tree2/α/ω-target', b'contents\n')])
    tree1.add(['α', 'ω-source'], ids=[a_id, rename_id])
    tree2.add(['α', 'α/ω-target'], ids=[a_id, rename_id])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.assertEqual([self.renamed(tree1, tree2, 'ω-source', 'α/ω-target', False)], self.do_iter_changes(tree1, tree2))
    self.assertEqualIterChanges([self.renamed(tree1, tree2, 'ω-source', 'α/ω-target', False)], self.do_iter_changes(tree1, tree2, specific_files=['α']))
    self.check_has_changes(True, tree1, tree2)
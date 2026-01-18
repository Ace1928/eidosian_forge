import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_renamed_and_unknown(self):
    """A file was moved on the filesystem, but not in bzr."""
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    root_id = tree1.path2id('')
    tree2.set_root_id(root_id)
    self.build_tree_contents([('tree1/a', b'a contents\n'), ('tree1/b', b'b contents\n'), ('tree2/a', b'a contents\n'), ('tree2/b', b'b contents\n')])
    tree1.add(['a', 'b'], ids=[b'a-id', b'b-id'])
    tree2.add(['a', 'b'], ids=[b'a-id', b'b-id'])
    os.rename('tree2/a', 'tree2/a2')
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_missing_in('a', tree2)
    expected = self.sorted([self.missing(b'a-id', 'a', 'a', tree2.path2id(''), 'file'), self.unversioned(tree2, 'a2')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, want_unversioned=True))
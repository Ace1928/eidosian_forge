import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_unknown_unicode(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    a_id = 'α-id'.encode()
    try:
        self.build_tree(['tree1/α/', 'tree2/α/', 'tree2/α/unknown_dir/', 'tree2/α/unknown_file', 'tree2/α/unknown_dir/file', 'tree2/ω-unknown_root_file'])
    except UnicodeError:
        raise tests.TestSkipped('Could not create Unicode files.')
    tree1.add(['α'], ids=[a_id])
    tree2.add(['α'], ids=[a_id])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    self.not_applicable_if_cannot_represent_unversioned(tree2)
    expected = self.sorted([self.unversioned(tree2, 'α/unknown_dir'), self.unversioned(tree2, 'α/unknown_file'), self.unversioned(tree2, 'ω-unknown_root_file')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, require_versioned=False, want_unversioned=True))
    self.assertEqual([], self.do_iter_changes(tree1, tree2))
    self.check_has_changes(False, tree1, tree2)
    expected = self.sorted([self.unversioned(tree2, 'α/unknown_dir'), self.unversioned(tree2, 'α/unknown_file')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['α'], require_versioned=False, want_unversioned=True))
    self.assertEqual([], self.do_iter_changes(tree1, tree2, specific_files=['α']))
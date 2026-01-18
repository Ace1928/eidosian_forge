import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_subdir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/c'])
    tree.add(['a', 'b', 'b/c'])
    tree.commit('initial')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/'), ('b/c', 'b/c')])
    a_contents = tree.get_file_text('a')
    tree.rename_one('a', 'b/d')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/c', 'b/c'), ('b/d', 'a')])
    self.assertPathDoesNotExist('a')
    self.assertFileEqual(a_contents, 'b/d')
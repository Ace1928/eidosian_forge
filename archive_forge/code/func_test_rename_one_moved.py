import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_moved(self):
    """Moving a moved entry works as expected."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a/', 'a/b', 'c/'])
    tree.add(['a', 'a/b', 'c'])
    tree.commit('initial')
    tree.rename_one('a/b', 'c/foo')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('c/', 'c/'), ('c/foo', 'a/b')])
    tree.rename_one('c/foo', 'bar')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a/', 'a/'), ('bar', 'a/b'), ('c/', 'c/')])
import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_samedir(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    a_contents = tree.get_file_text('a')
    tree.rename_one('a', 'foo')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('foo', 'a')])
    self.assertPathDoesNotExist('a')
    self.assertFileEqual(a_contents, 'foo')
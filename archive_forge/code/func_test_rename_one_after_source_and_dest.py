import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_after_source_and_dest(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b/', 'b/foo'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    with open('a') as a_file:
        a_text = a_file.read()
    with open('b/foo') as foo_file:
        foo_text = foo_file.read()
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/')])
    self.assertRaises(errors.RenameFailedFilesExist, tree.rename_one, 'a', 'b/foo', after=False)
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b/', 'b/')])
    self.assertFileEqual(a_text, 'a')
    self.assertFileEqual(foo_text, 'b/foo')
    tree.rename_one('a', 'b/foo', after=True)
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('b/', 'b/'), ('b/foo', 'a')])
    self.assertFileEqual(a_text, 'a')
    self.assertFileEqual(foo_text, 'b/foo')
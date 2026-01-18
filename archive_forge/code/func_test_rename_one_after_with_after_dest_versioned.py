import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_one_after_with_after_dest_versioned(self):
    """ using after with an already versioned file should fail """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add(['a', 'b'])
    tree.commit('initial')
    os.unlink('a')
    self.assertPathRelations(tree.basis_tree(), tree, [('', ''), ('a', 'a'), ('b', 'b')])
    e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', 'b', after=True)
    self.assertIsInstance(e.extra, errors.AlreadyVersionedError)
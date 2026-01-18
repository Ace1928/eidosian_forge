import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_nonempty_basis_to_dirstate_tree(self):
    tree = self.make_workingtree()
    tree.commit('first post')
    tree.lock_read()
    basis_tree = tree.basis_tree()
    basis_tree.lock_read()
    optimiser = InterTree.get(basis_tree, tree)
    tree.unlock()
    basis_tree.unlock()
    self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)
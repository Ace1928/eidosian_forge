import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_tree_to_basis_in_other_tree(self):
    tree = self.make_workingtree('a')
    tree.commit('first post')
    tree2 = self.make_workingtree('b')
    tree2.pull(tree.branch)
    basis_tree = tree.basis_tree()
    tree2.lock_read()
    basis_tree.lock_read()
    optimiser = InterTree.get(basis_tree, tree2)
    tree2.unlock()
    basis_tree.unlock()
    self.assertIsInstance(optimiser, workingtree_4.InterDirStateTree)
import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_revtree_to_revtree_not_interdirstate(self):
    tree = self.make_workingtree()
    rev_id = tree.commit('first post')
    rev_id2 = tree.commit('second post')
    rev_tree = tree.branch.repository.revision_tree(rev_id)
    rev_tree2 = tree.branch.repository.revision_tree(rev_id2)
    optimiser = InterTree.get(rev_tree, rev_tree2)
    self.assertIsInstance(optimiser, InterTree)
    self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))
    optimiser = InterTree.get(rev_tree2, rev_tree)
    self.assertIsInstance(optimiser, InterTree)
    self.assertFalse(isinstance(optimiser, workingtree_4.InterDirStateTree))
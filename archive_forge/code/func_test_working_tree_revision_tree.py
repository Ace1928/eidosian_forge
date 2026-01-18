from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_working_tree_revision_tree(self):
    tree = self.make_branch_and_tree('.')
    rev_id = tree.commit('first post')
    rev_tree = tree.branch.repository.revision_tree(rev_id)
    optimiser = InterTree.get(rev_tree, tree)
    self.assertIsInstance(optimiser, InterTree)
    optimiser = InterTree.get(tree, rev_tree)
    self.assertIsInstance(optimiser, InterTree)
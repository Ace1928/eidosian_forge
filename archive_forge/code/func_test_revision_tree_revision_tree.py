from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_revision_tree_revision_tree(self):
    tree = self.make_branch_and_tree('.')
    rev_id = tree.commit('first post')
    rev_id2 = tree.commit('second post', allow_pointless=True)
    rev_tree = tree.branch.repository.revision_tree(rev_id)
    rev_tree2 = tree.branch.repository.revision_tree(rev_id2)
    optimiser = InterTree.get(rev_tree, rev_tree2)
    self.assertIsInstance(optimiser, InterTree)
    optimiser = InterTree.get(rev_tree2, rev_tree)
    self.assertIsInstance(optimiser, InterTree)
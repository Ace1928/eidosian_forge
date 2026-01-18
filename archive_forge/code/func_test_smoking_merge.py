import os
from breezy import conflicts, errors, merge
from breezy.tests import per_workingtree
from breezy.workingtree import PointlessMerge
def test_smoking_merge(self):
    """Smoke test of merge_from_branch."""
    self.create_two_trees_for_merging()
    self.tree_to.merge_from_branch(self.tree_from.branch)
    self.assertEqual([self.to_second_rev, self.second_rev], self.tree_to.get_parent_ids())
import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def assertEqualIterChanges(self, left_changes, right_changes):
    """Assert that left_changes == right_changes.

        :param left_changes: A list of the output from iter_changes.
        :param right_changes: A list of the output from iter_changes.
        """
    left_changes = self.sorted(left_changes)
    right_changes = self.sorted(right_changes)
    if left_changes == right_changes:
        return
    left_dict = {item[0]: item for item in left_changes}
    right_dict = {item[0]: item for item in right_changes}
    if len(left_dict) != len(left_changes) or len(right_dict) != len(right_changes):
        self.assertEqual(left_changes, right_changes)
    keys = set(left_dict).union(set(right_dict))
    different = []
    same = []
    for key in keys:
        left_item = left_dict.get(key)
        right_item = right_dict.get(key)
        if left_item == right_item:
            same.append(str(left_item))
        else:
            different.append(' {}\n {}'.format(left_item, right_item))
    self.fail('iter_changes output different. Unchanged items:\n' + '\n'.join(same) + '\nChanged items:\n' + '\n'.join(different))
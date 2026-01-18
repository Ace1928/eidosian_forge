import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_smart_add_symlink_pointing_outside(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/link@', '../../../../target')])
    tree.smart_add(['tree/link'])
    self.assertTrue(tree.is_versioned('link'))
    self.assertFalse(tree.is_versioned('target'))
    self.assertEqual('symlink', tree.kind('link'))
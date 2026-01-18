import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_pending_changes(self):
    tree = self.make_test_tree()
    self.build_tree_contents([('debian/changelog', 'blah')])
    with tree.lock_write():
        self.assertRaises(WorkspaceDirty, check_clean_tree, tree)
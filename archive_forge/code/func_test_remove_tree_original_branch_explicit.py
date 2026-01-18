import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_original_branch_explicit(self):
    self.run_bzr('remove-tree branch1')
    self.assertPathDoesNotExist('branch1/foo')
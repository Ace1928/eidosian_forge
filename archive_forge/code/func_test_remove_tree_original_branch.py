import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_original_branch(self):
    self.run_bzr('remove-tree', working_dir='branch1')
    self.assertPathDoesNotExist('branch1/foo')
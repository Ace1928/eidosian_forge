import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_checkout(self):
    self.tree.branch.create_checkout('branch2', lightweight=False)
    self.assertPathExists('branch2/foo')
    self.run_bzr('remove-tree', working_dir='branch2')
    self.assertPathDoesNotExist('branch2/foo')
    self.assertPathExists('branch1/foo')
import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_sprouted_branch_explicit(self):
    self.tree.controldir.sprout('branch2')
    self.assertPathExists('branch2/foo')
    self.run_bzr('remove-tree branch2')
    self.assertPathDoesNotExist('branch2/foo')
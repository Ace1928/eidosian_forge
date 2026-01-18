import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_repeatedly(self):
    self.run_bzr('remove-tree branch1')
    self.assertPathDoesNotExist('branch1/foo')
    output = self.run_bzr_error(['No working tree to remove'], 'remove-tree branch1', retcode=3)
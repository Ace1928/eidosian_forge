import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_empty_dir(self):
    os.mkdir('branch2')
    output = self.run_bzr_error(['Not a branch'], 'remove-tree', retcode=3, working_dir='branch2')
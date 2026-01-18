import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_shelve_in_subdir(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file', 'tree/dir/'])
    tree.add('file')
    os.chdir('tree/dir')
    self.run_bzr('shelve --all ../file')
import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_backslash(self):
    if os.path.sep == '\\':
        raise TestNotApplicable('unable to add filenames with backslashes where  it is the path separator')
    tree = self.make_branch_and_tree('.')
    self.build_tree(['\\'])
    self.assertEqual('adding \\\n', self.run_bzr('add \\\\')[0])
    self.assertEqual('\\\n', self.run_bzr('ls --versioned')[0])
    self.assertEqual('', self.run_bzr('rm \\\\')[0])
    self.assertEqual('', self.run_bzr('ls --versioned')[0])
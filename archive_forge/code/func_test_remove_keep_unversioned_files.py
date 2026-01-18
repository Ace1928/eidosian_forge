import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_keep_unversioned_files(self):
    self.build_tree(files)
    tree = self.make_branch_and_tree('.')
    self.run_bzr('remove --keep a', error_regexes=['a is not versioned.'])
    self.assertFilesUnversioned(files)
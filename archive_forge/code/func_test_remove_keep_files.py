import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_keep_files(self):
    tree = self._make_tree_and_add(files)
    self.run_bzr("commit -m 'added files'")
    self.run_bzr('remove --keep a b b/c d', error_regexes=['removed a', 'removed b', 'removed b/c', 'removed d'])
    self.assertFilesUnversioned(files)
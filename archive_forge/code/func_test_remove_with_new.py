import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_with_new(self):
    tree = self._make_tree_and_add(files)
    self.run_bzr('remove --new --keep', error_regexes=['removed a', 'removed b', 'removed b/c'])
    self.assertFilesUnversioned(files)
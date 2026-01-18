import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_files(self):
    tree = self._make_tree_and_add(files)
    self.run_bzr("commit -m 'added files'")
    self.run_bzr('remove a b b/c d', error_regexes=['deleted a', 'deleted b', 'deleted b/c', 'deleted d'])
    self.assertFilesDeleted(files)
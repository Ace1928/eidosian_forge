import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_keep_one_file(self):
    tree = self._make_tree_and_add([a])
    self.run_bzr('remove --keep a', error_regexes=['removed a'])
    self.assertFilesUnversioned([a])
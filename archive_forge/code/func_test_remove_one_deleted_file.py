import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_one_deleted_file(self):
    tree = self._make_tree_and_add([a])
    self.run_bzr("commit -m 'added a'")
    os.unlink(a)
    self.assertInWorkingTree(a)
    self.run_bzr('remove a')
    self.assertNotInWorkingTree(a)
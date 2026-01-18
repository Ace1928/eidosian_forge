import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_changed_ignored_files(self):
    tree = self._make_tree_and_add(['a'])
    self.run_bzr(['ignore', 'a'])
    self.run_bzr_remove_changed_files(['a'])
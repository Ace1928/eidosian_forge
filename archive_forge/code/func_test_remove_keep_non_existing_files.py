import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_keep_non_existing_files(self):
    tree = self._make_tree_and_add([])
    self.run_bzr('remove --keep b', error_regexes=['b is not versioned.'])
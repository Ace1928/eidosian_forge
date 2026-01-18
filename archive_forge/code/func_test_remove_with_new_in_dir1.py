import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def test_remove_with_new_in_dir1(self):
    tree = self._make_tree_and_add(files)
    self.run_bzr('remove --new --keep b b/c', error_regexes=['removed b', 'removed b/c'])
    tree = WorkingTree.open('.')
    self.assertInWorkingTree(a)
    self.assertEqual(tree.path2id(a), a.encode('utf-8') + _id)
    self.assertFilesUnversioned([b, c])
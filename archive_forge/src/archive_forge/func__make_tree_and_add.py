import os
import sys
from breezy import osutils
from breezy.tests import (TestCaseWithTransport, TestNotApplicable,
from breezy.workingtree import WorkingTree
def _make_tree_and_add(self, paths):
    tree = self.make_branch_and_tree('.')
    with tree.lock_write():
        self.build_tree(paths)
        for path in paths:
            file_id = path.replace('/', '_').encode('utf-8') + _id
            tree.add(path, ids=file_id)
    return tree
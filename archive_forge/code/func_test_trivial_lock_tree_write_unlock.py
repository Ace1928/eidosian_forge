import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_trivial_lock_tree_write_unlock(self):
    """Locking for tree write is ok when the branch is not locked."""
    wt = self.make_branch_and_tree('.')
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
    wt.lock_tree_write()
    try:
        self.assertTrue(wt.is_locked())
        self.assertTrue(wt.branch.is_locked())
    finally:
        wt.unlock()
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
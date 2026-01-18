import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_trivial_lock_tree_write_branch_read_locked(self):
    """It is ok to lock_tree_write when the branch is read locked."""
    wt = self.make_branch_and_tree('.')
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
    wt.branch.lock_read()
    try:
        wt.lock_tree_write()
    except errors.ReadOnlyError:
        wt.branch.unlock()
        self.assertFalse(wt.is_locked())
        self.assertFalse(wt.branch.is_locked())
        return
    try:
        self.assertTrue(wt.is_locked())
        self.assertTrue(wt.branch.is_locked())
    finally:
        wt.unlock()
    self.assertFalse(wt.is_locked())
    self.assertTrue(wt.branch.is_locked())
    wt.branch.unlock()
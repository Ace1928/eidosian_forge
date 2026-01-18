import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_trivial_lock_read_unlock(self):
    """Locking and unlocking should work trivially."""
    wt = self.make_branch_and_tree('.')
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
    wt.lock_read()
    try:
        self.assertTrue(wt.is_locked())
        self.assertTrue(wt.branch.is_locked())
    finally:
        wt.unlock()
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
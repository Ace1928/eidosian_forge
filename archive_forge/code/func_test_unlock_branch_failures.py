import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unlock_branch_failures(self):
    """If the branch unlock fails the tree must still unlock."""
    wt = self.make_branch_and_tree('.')
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
    wt.lock_write()
    self.assertTrue(wt.is_locked())
    self.assertTrue(wt.branch.is_locked())
    wt.branch.unlock()
    self.assertRaises(errors.LockNotHeld, wt.unlock)
    self.assertFalse(wt.is_locked())
    self.assertFalse(wt.branch.is_locked())
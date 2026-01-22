import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
class TestWorkingTreeLocking(TestCaseWithWorkingTree):

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

    def test_lock_read_returns_unlocker(self):
        wt = self.make_branch_and_tree('.')
        self.assertThat(wt.lock_read, ReturnsUnlockable(wt))

    def test_trivial_lock_write_unlock(self):
        """Locking for write and unlocking should work trivially."""
        wt = self.make_branch_and_tree('.')
        self.assertFalse(wt.is_locked())
        self.assertFalse(wt.branch.is_locked())
        wt.lock_write()
        try:
            self.assertTrue(wt.is_locked())
            self.assertTrue(wt.branch.is_locked())
        finally:
            wt.unlock()
        self.assertFalse(wt.is_locked())
        self.assertFalse(wt.branch.is_locked())

    def test_lock_write_returns_unlocker(self):
        wt = self.make_branch_and_tree('.')
        self.assertThat(wt.lock_write, ReturnsUnlockable(wt))

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

    def test_lock_tree_write_returns_unlocker(self):
        wt = self.make_branch_and_tree('.')
        self.assertThat(wt.lock_tree_write, ReturnsUnlockable(wt))

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

    def _test_unlock_with_lock_method(self, methodname):
        """Create a tree and then test its unlocking behaviour.

        :param methodname: The lock method to use to establish locks.
        """
        if sys.platform == 'win32':
            raise TestSkipped("don't use oslocks on win32 in unix manner")
        self.thisFailsStrictLockCheck()
        tree = self.make_branch_and_tree('tree')
        getattr(tree, methodname)()
        getattr(tree, methodname)()
        if tree.supports_file_ids:
            old_root = tree.path2id('')
        tree.add('')
        reference_tree = tree.controldir.open_workingtree()
        if tree.supports_file_ids:
            self.assertEqual(old_root, reference_tree.path2id(''))
        tree.unlock()
        reference_tree = tree.controldir.open_workingtree()
        if tree.supports_file_ids:
            self.assertEqual(old_root, reference_tree.path2id(''))
        tree.unlock()
        reference_tree = tree.controldir.open_workingtree()
        if reference_tree.supports_file_ids:
            self.assertIsNot(None, reference_tree.path2id(''))
        self.assertTrue(reference_tree.is_versioned(''))

    def test_unlock_from_tree_write_lock_flushes(self):
        self._test_unlock_with_lock_method('lock_tree_write')

    def test_unlock_from_write_lock_flushes(self):
        self._test_unlock_with_lock_method('lock_write')

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

    def test_failing_to_lock_branch_does_not_lock(self):
        """If the branch cannot be locked, dont lock the tree."""
        wt = self.make_branch_and_tree('.')
        branch_copy = branch.Branch.open('.')
        branch_copy.lock_write()
        try:
            try:
                wt.lock_read()
            except errors.LockError:
                self.assertFalse(wt.is_locked())
                self.assertFalse(wt.branch.is_locked())
                return
            else:
                wt.unlock()
        finally:
            branch_copy.unlock()

    def test_failing_to_lock_write_branch_does_not_lock(self):
        """If the branch cannot be write locked, dont lock the tree."""
        wt = self.make_branch_and_tree('.')
        branch_copy = branch.Branch.open('.')
        branch_copy.lock_write()
        try:
            try:
                self.assertRaises(errors.LockError, wt.lock_write)
                self.assertFalse(wt.is_locked())
                self.assertFalse(wt.branch.is_locked())
            finally:
                if wt.is_locked():
                    wt.unlock()
        finally:
            branch_copy.unlock()

    def test_failing_to_lock_tree_write_branch_does_not_lock(self):
        """If the branch cannot be read locked, dont lock the tree."""
        wt = self.make_branch_and_tree('.')
        branch_copy = branch.Branch.open('.')
        branch_copy.lock_write()
        try:
            try:
                wt.lock_tree_write()
            except errors.LockError:
                self.assertFalse(wt.is_locked())
                self.assertFalse(wt.branch.is_locked())
                return
            else:
                wt.unlock()
        finally:
            branch_copy.unlock()
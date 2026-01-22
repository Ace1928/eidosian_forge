from breezy import errors, tests
from breezy.tests import lock_helpers, per_branch
from breezy.tests.matchers import *
class TestBranchLocking(per_branch.TestCaseWithBranch):

    def setUp(self):
        super().setUp()
        self.reduceLockdirTimeout()

    def get_instrumented_branch(self):
        """Get a Branch object which has been instrumented"""
        self.locks = []
        b = lock_helpers.LockWrapper(self.locks, self.get_branch(), 'b')
        b.repository = lock_helpers.LockWrapper(self.locks, b.repository, 'r')
        bcf = getattr(b, 'control_files', None)
        rcf = getattr(b.repository, 'control_files', None)
        if rcf is None:
            self.combined_branch = False
        else:
            self.combined_control = bcf is rcf and bcf is not None
        try:
            b.control_files = lock_helpers.LockWrapper(self.locks, b.control_files, 'bc')
        except AttributeError:
            raise tests.TestSkipped('Could not instrument branch control files.')
        if self.combined_control:
            b.repository.control_files = lock_helpers.LockWrapper(self.locks, b.repository.control_files, 'rc')
        return b

    def test_01_lock_read(self):
        b = self.get_instrumented_branch()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        b.lock_read()
        try:
            self.assertTrue(b.is_locked())
            self.assertTrue(b.repository.is_locked())
        finally:
            b.unlock()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        if self.combined_control:
            self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('rc', 'lr', True), ('bc', 'lr', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
        else:
            self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('bc', 'lr', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True)], self.locks)

    def test_02_lock_write(self):
        b = self.get_instrumented_branch()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        b.lock_write()
        try:
            self.assertTrue(b.is_locked())
            self.assertTrue(b.repository.is_locked())
        finally:
            b.unlock()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        if self.combined_control:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
        else:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', True)], self.locks)

    def test_03_lock_fail_unlock_repo(self):
        b = self.get_instrumented_branch()
        b.repository.disable_unlock()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        b.lock_write()
        try:
            self.assertTrue(b.is_locked())
            self.assertTrue(b.repository.is_locked())
            self.assertLogsError(lock_helpers.TestPreventLocking, b.unlock)
            if self.combined_control:
                self.assertTrue(b.is_locked())
            else:
                self.assertFalse(b.is_locked())
            self.assertTrue(b.repository.is_locked())
            if self.combined_control:
                self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', False)], self.locks)
            else:
                self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', True), ('r', 'ul', False)], self.locks)
        finally:
            b.repository._other.unlock()

    def test_04_lock_fail_unlock_control(self):
        b = self.get_instrumented_branch()
        b.control_files.disable_unlock()
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        b.lock_write()
        try:
            self.assertTrue(b.is_locked())
            self.assertTrue(b.repository.is_locked())
            self.assertLogsError(lock_helpers.TestPreventLocking, b.unlock)
            self.assertTrue(b.is_locked())
            self.assertTrue(b.repository.is_locked())
            if self.combined_control:
                self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', False), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
            else:
                self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', True), ('b', 'ul', True), ('bc', 'ul', False)], self.locks)
        finally:
            b.control_files._other.unlock()

    def test_05_lock_read_fail_repo(self):
        b = self.get_instrumented_branch()
        b.repository.disable_lock_read()
        self.assertRaises(lock_helpers.TestPreventLocking, b.lock_read)
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        self.assertEqual([('b', 'lr', True), ('r', 'lr', False)], self.locks)

    def test_06_lock_write_fail_repo(self):
        b = self.get_instrumented_branch()
        b.repository.disable_lock_write()
        self.assertRaises(lock_helpers.TestPreventLocking, b.lock_write)
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        self.assertEqual([('b', 'lw', True), ('r', 'lw', False)], self.locks)

    def test_07_lock_read_fail_control(self):
        b = self.get_instrumented_branch()
        b.control_files.disable_lock_read()
        self.assertRaises(lock_helpers.TestPreventLocking, b.lock_read)
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        if self.combined_control:
            self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('rc', 'lr', True), ('bc', 'lr', False), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
        else:
            self.assertEqual([('b', 'lr', True), ('r', 'lr', True), ('bc', 'lr', False), ('r', 'ul', True)], self.locks)

    def test_08_lock_write_fail_control(self):
        b = self.get_instrumented_branch()
        b.control_files.disable_lock_write()
        self.assertRaises(lock_helpers.TestPreventLocking, b.lock_write)
        self.assertFalse(b.is_locked())
        self.assertFalse(b.repository.is_locked())
        if self.combined_control:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('rc', 'lw', True), ('bc', 'lw', False), ('r', 'ul', True), ('rc', 'ul', True)], self.locks)
        else:
            self.assertEqual([('b', 'lw', True), ('r', 'lw', True), ('bc', 'lw', False), ('r', 'ul', True)], self.locks)

    def test_lock_write_returns_None_refuses_token(self):
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is not None:
                return
            self.assertRaises(errors.TokenLockingNotSupported, branch.lock_write, token='token')

    def test_reentering_lock_write_raises_on_token_mismatch(self):
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is None:
                return
            different_branch_token = lock.token + b'xxx'
            self.assertRaises(errors.TokenMismatch, branch.lock_write, token=different_branch_token)

    def test_lock_write_with_nonmatching_token(self):
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is None:
                return
            different_branch_token = lock.token + b'xxx'
            new_branch = branch.controldir.open_branch()
            new_branch.repository = branch.repository
            self.assertRaises(errors.TokenMismatch, new_branch.lock_write, token=different_branch_token)

    def test_lock_write_with_matching_token(self):
        """Test that a branch can be locked with a token, if it is already
        locked by that token."""
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is None:
                return
            branch.lock_write(token=lock.token)
            branch.unlock()
            new_branch = branch.controldir.open_branch()
            new_branch.repository = branch.repository
            new_branch.lock_write(token=lock.token)
            new_branch.unlock()

    def test_unlock_after_lock_write_with_token(self):
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is None:
                return
            new_branch = branch.controldir.open_branch()
            new_branch.repository = branch.repository
            new_branch.lock_write(token=lock.token)
            new_branch.unlock()
            self.assertTrue(branch.get_physical_lock_status())

    def test_lock_write_with_token_fails_when_unlocked(self):
        branch = self.make_branch('b')
        token = branch.lock_write().token
        branch.unlock()
        if token is None:
            return
        self.assertRaises(errors.TokenMismatch, branch.lock_write, token=token)

    def test_lock_write_reenter_with_token(self):
        branch = self.make_branch('b')
        with branch.lock_write() as lock:
            if lock.token is None:
                return
            branch.lock_write(token=lock.token)
            branch.unlock()
        new_branch = branch.controldir.open_branch()
        new_branch.lock_write()
        new_branch.unlock()

    def test_leave_lock_in_place(self):
        branch = self.make_branch('b')
        token = None
        with branch.lock_write() as lock:
            token = lock.token
            if lock.token is None:
                self.assertRaises(NotImplementedError, branch.leave_lock_in_place)
                return
            branch.leave_lock_in_place()
        self.assertRaises(errors.LockContention, branch.lock_write)
        branch.lock_write(token)
        branch.dont_leave_lock_in_place()
        branch.unlock()

    def test_dont_leave_lock_in_place(self):
        branch = self.make_branch('b')
        token = branch.lock_write().token
        try:
            if token is None:
                self.assertRaises(NotImplementedError, branch.dont_leave_lock_in_place)
                return
            try:
                branch.leave_lock_in_place()
            except NotImplementedError:
                return
            try:
                branch.repository.leave_lock_in_place()
            except NotImplementedError:
                repo_token = None
            else:
                repo_token = branch.repository.lock_write()
                branch.repository.unlock()
        finally:
            branch.unlock()
        new_branch = branch.controldir.open_branch()
        if repo_token is not None:
            new_branch.repository.lock_write(token=repo_token)
        new_branch.lock_write(token=token)
        if repo_token is not None:
            new_branch.repository.unlock()
        new_branch.dont_leave_lock_in_place()
        if repo_token is not None:
            new_branch.repository.dont_leave_lock_in_place()
        new_branch.unlock()
        branch.lock_write()
        branch.unlock()

    def test_lock_read_then_unlock(self):
        branch = self.make_branch('b')
        branch.lock_read()
        branch.unlock()

    def test_lock_read_context_manager(self):
        branch = self.make_branch('b')
        self.assertFalse(branch.is_locked())
        with branch.lock_read():
            self.assertTrue(branch.is_locked())

    def test_lock_read_returns_unlockable(self):
        branch = self.make_branch('b')
        self.assertThat(branch.lock_read, ReturnsUnlockable(branch))

    def test_lock_write_locks_repo_too(self):
        branch = self.make_branch('b')
        branch = branch.controldir.open_branch()
        branch.lock_write()
        try:
            self.assertTrue(branch.repository.is_write_locked())
            if not branch.repository.get_physical_lock_status():
                return
            new_repo = branch.controldir.open_repository()
            self.assertRaises(errors.LockContention, new_repo.lock_write)
            branch.repository.lock_write()
            branch.repository.unlock()
        finally:
            branch.unlock()

    def test_lock_write_returns_unlockable(self):
        branch = self.make_branch('b')
        self.assertThat(branch.lock_write, ReturnsUnlockable(branch))

    def test_lock_write_raises_in_lock_read(self):
        branch = self.make_branch('b')
        branch.lock_read()
        self.addCleanup(branch.unlock)
        err = self.assertRaises(errors.ReadOnlyError, branch.lock_write)

    def test_lock_and_unlock_leaves_repo_unlocked(self):
        branch = self.make_branch('b')
        branch.lock_write()
        branch.unlock()
        self.assertRaises(errors.LockNotHeld, branch.repository.unlock)
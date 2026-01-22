import breezy
from .. import errors, lockdir, osutils, transport
from ..bzr.tests.test_smart import TestCaseWithSmartMedium
from ..lockable_files import LockableFiles, TransportLock
from ..transactions import (PassThroughTransaction, ReadOnlyTransaction,
from . import TestCaseInTempDir, TestNotApplicable
from .test_transactions import DummyWeave
class _TestLockableFiles_mixin:

    def test_transactions(self):
        self.assertIs(self.lockable.get_transaction().__class__, PassThroughTransaction)
        self.lockable.lock_read()
        try:
            self.assertIs(self.lockable.get_transaction().__class__, ReadOnlyTransaction)
        finally:
            self.lockable.unlock()
        self.assertIs(self.lockable.get_transaction().__class__, PassThroughTransaction)
        self.lockable.lock_write()
        self.assertIs(self.lockable.get_transaction().__class__, WriteTransaction)
        vf = DummyWeave('a')
        self.lockable.get_transaction().register_dirty(vf)
        self.lockable.unlock()
        self.assertTrue(vf.finished)

    def test__escape(self):
        self.assertEqual('%25', self.lockable._escape('%'))

    def test__escape_empty(self):
        self.assertEqual('', self.lockable._escape(''))

    def test_break_lock(self):
        self.lockable.lock_write()
        try:
            self.assertRaises(AssertionError, self.lockable.break_lock)
        except NotImplementedError:
            self.lockable.unlock()
            raise TestNotApplicable('{!r} is not breakable'.format(self.lockable))
        l2 = self.get_lockable()
        orig_factory = breezy.ui.ui_factory
        breezy.ui.ui_factory = breezy.ui.CannedInputUIFactory([True])
        try:
            l2.break_lock()
        finally:
            breezy.ui.ui_factory = orig_factory
        try:
            l2.lock_write()
            l2.unlock()
        finally:
            self.assertRaises(errors.LockBroken, self.lockable.unlock)
            self.assertFalse(self.lockable.is_locked())

    def test_lock_write_returns_None_refuses_token(self):
        token = self.lockable.lock_write()
        self.addCleanup(self.lockable.unlock)
        if token is not None:
            raise TestNotApplicable('{!r} uses tokens'.format(self.lockable))
        self.assertRaises(errors.TokenLockingNotSupported, self.lockable.lock_write, token='token')

    def test_lock_write_returns_token_when_given_token(self):
        token = self.lockable.lock_write()
        self.addCleanup(self.lockable.unlock)
        if token is None:
            return
        new_lockable = self.get_lockable()
        token_from_new_lockable = new_lockable.lock_write(token=token)
        self.addCleanup(new_lockable.unlock)
        self.assertEqual(token, token_from_new_lockable)

    def test_lock_write_raises_on_token_mismatch(self):
        token = self.lockable.lock_write()
        self.addCleanup(self.lockable.unlock)
        if token is None:
            return
        different_token = token + b'xxx'
        self.assertRaises(errors.TokenMismatch, self.lockable.lock_write, token=different_token)
        new_lockable = self.get_lockable()
        self.assertRaises(errors.TokenMismatch, new_lockable.lock_write, token=different_token)

    def test_lock_write_with_matching_token(self):
        token = self.lockable.lock_write()
        self.addCleanup(self.lockable.unlock)
        if token is None:
            return
        self.lockable.lock_write(token=token)
        self.lockable.unlock()
        new_lockable = self.get_lockable()
        new_lockable.lock_write(token=token)
        new_lockable.unlock()

    def test_unlock_after_lock_write_with_token(self):
        token = self.lockable.lock_write()
        self.addCleanup(self.lockable.unlock)
        if token is None:
            return
        new_lockable = self.get_lockable()
        new_lockable.lock_write(token=token)
        new_lockable.unlock()
        self.assertTrue(self.lockable.get_physical_lock_status())

    def test_lock_write_with_token_fails_when_unlocked(self):
        token = self.lockable.lock_write()
        self.lockable.unlock()
        if token is None:
            return
        self.assertRaises(errors.TokenMismatch, self.lockable.lock_write, token=token)

    def test_lock_write_reenter_with_token(self):
        token = self.lockable.lock_write()
        try:
            if token is None:
                return
            token_from_reentry = self.lockable.lock_write(token=token)
            try:
                self.assertEqual(token, token_from_reentry)
            finally:
                self.lockable.unlock()
        finally:
            self.lockable.unlock()
        new_lockable = self.get_lockable()
        new_lockable.lock_write()
        new_lockable.unlock()

    def test_second_lock_write_returns_same_token(self):
        first_token = self.lockable.lock_write()
        try:
            if first_token is None:
                return
            second_token = self.lockable.lock_write()
            try:
                self.assertEqual(first_token, second_token)
            finally:
                self.lockable.unlock()
        finally:
            self.lockable.unlock()

    def test_leave_in_place(self):
        token = self.lockable.lock_write()
        try:
            if token is None:
                return
            self.lockable.leave_in_place()
        finally:
            self.lockable.unlock()
        self.assertRaises(errors.LockContention, self.lockable.lock_write)
        self.lockable.lock_write(token=token)
        self.lockable.unlock()
        self.lockable.lock_write(token=token)
        self.lockable.dont_leave_in_place()
        self.lockable.unlock()

    def test_dont_leave_in_place(self):
        token = self.lockable.lock_write()
        try:
            if token is None:
                return
            self.lockable.leave_in_place()
        finally:
            self.lockable.unlock()
        new_lockable = self.get_lockable()
        new_lockable.lock_write(token=token)
        new_lockable.dont_leave_in_place()
        new_lockable.unlock()
        third_lockable = self.get_lockable()
        third_lockable.lock_write()
        third_lockable.unlock()
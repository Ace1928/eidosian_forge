from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
class TestDummyLock(TestCase):

    def test_lock_initially_not_held(self):
        l = DummyLock()
        self.assertFalse(l.is_locked())

    def test_lock_not_reentrant(self):
        l = DummyLock()
        l.lock_read()
        self.assertRaises(LockError, l.lock_read)

    def test_detect_underlock(self):
        l = DummyLock()
        self.assertRaises(LockError, l.unlock)

    def test_basic_locking(self):
        real_lock = DummyLock()
        self.assertFalse(real_lock.is_locked())
        real_lock.lock_read()
        self.assertTrue(real_lock.is_locked())
        real_lock.unlock()
        self.assertFalse(real_lock.is_locked())
        result = real_lock.lock_write()
        self.assertEqual('token', result)
        self.assertTrue(real_lock.is_locked())
        real_lock.unlock()
        self.assertFalse(real_lock.is_locked())
        self.assertEqual(['lock_read', 'unlock', 'lock_write', 'unlock'], real_lock._calls)

    def test_break_lock(self):
        l = DummyLock()
        l.lock_write()
        l.break_lock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_write', 'break'], l._calls)
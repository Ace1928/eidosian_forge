from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
class TestCountedLock(TestCase):

    def test_read_lock(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertFalse(l.is_locked())
        l.lock_read()
        l.lock_read()
        self.assertTrue(l.is_locked())
        l.unlock()
        self.assertTrue(l.is_locked())
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_read', 'unlock'], real_lock._calls)

    def test_unlock_not_locked(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertRaises(LockNotHeld, l.unlock)

    def test_read_lock_while_write_locked(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_write()
        l.lock_read()
        self.assertEqual('token', l.lock_write())
        l.unlock()
        l.unlock()
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_write', 'unlock'], real_lock._calls)

    def test_write_lock_while_read_locked(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_read()
        self.assertRaises(ReadOnlyError, l.lock_write)
        self.assertRaises(ReadOnlyError, l.lock_write)
        l.unlock()
        self.assertFalse(l.is_locked())
        self.assertEqual(['lock_read', 'unlock'], real_lock._calls)

    def test_write_lock_reentrant(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        self.assertEqual('token', l.lock_write())
        self.assertEqual('token', l.lock_write())
        l.unlock()
        l.unlock()

    def test_reenter_with_token(self):
        real_lock = DummyLock()
        l1 = CountedLock(real_lock)
        l2 = CountedLock(real_lock)
        token = l1.lock_write()
        self.assertEqual('token', token)
        del l1
        self.assertTrue(real_lock.is_locked())
        self.assertFalse(l2.is_locked())
        self.assertEqual(token, l2.lock_write(token=token))
        self.assertTrue(l2.is_locked())
        self.assertTrue(real_lock.is_locked())
        l2.unlock()
        self.assertFalse(l2.is_locked())
        self.assertFalse(real_lock.is_locked())

    def test_break_lock(self):
        real_lock = DummyLock()
        l = CountedLock(real_lock)
        l.lock_write()
        l.lock_write()
        self.assertTrue(real_lock.is_locked())
        l.break_lock()
        self.assertFalse(l.is_locked())
        self.assertFalse(real_lock.is_locked())
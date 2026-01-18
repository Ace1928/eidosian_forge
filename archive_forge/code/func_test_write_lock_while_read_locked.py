from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_write_lock_while_read_locked(self):
    real_lock = DummyLock()
    l = CountedLock(real_lock)
    l.lock_read()
    self.assertRaises(ReadOnlyError, l.lock_write)
    self.assertRaises(ReadOnlyError, l.lock_write)
    l.unlock()
    self.assertFalse(l.is_locked())
    self.assertEqual(['lock_read', 'unlock'], real_lock._calls)
from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
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
from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_write_lock_reentrant(self):
    real_lock = DummyLock()
    l = CountedLock(real_lock)
    self.assertEqual('token', l.lock_write())
    self.assertEqual('token', l.lock_write())
    l.unlock()
    l.unlock()
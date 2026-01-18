from ..counted_lock import CountedLock
from ..errors import LockError, LockNotHeld, ReadOnlyError, TokenMismatch
from . import TestCase
def test_unlock_not_locked(self):
    real_lock = DummyLock()
    l = CountedLock(real_lock)
    self.assertRaises(LockNotHeld, l.unlock)
from __future__ import absolute_import
from twisted.trial.unittest import TestCase
from .._util import synchronized
class SynchronizedTests(TestCase):
    """
    Tests for the synchronized decorator.
    """

    def test_return(self):
        """
        A method wrapped with @synchronized is called with the lock acquired,
        and it is released on return.
        """
        obj = Lockable()
        self.assertEqual(obj.check(1, y=2), (1, 2))
        self.assertFalse(obj._lock.locked)

    def test_raise(self):
        """
        A method wrapped with @synchronized is called with the lock acquired,
        and it is released on exception raise.
        """
        obj = Lockable()
        self.assertRaises(ZeroDivisionError, obj.raiser)
        self.assertFalse(obj._lock.locked)

    def test_name(self):
        """
        A method wrapped with @synchronized preserves its name.
        """
        self.assertEqual(Lockable.check.__name__, 'check')

    def test_marked(self):
        """
        A method wrapped with @synchronized is marked as synchronized.
        """
        self.assertEqual(Lockable.check.synchronized, True)
from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class DeferLaterTests(unittest.TestCase):
    """
    Tests for L{task.deferLater}.
    """

    def test_callback(self):
        """
        The L{Deferred} returned by L{task.deferLater} is called back after
        the specified delay with the result of the function passed in.
        """
        results = []
        flag = object()

        def callable(foo, bar):
            results.append((foo, bar))
            return flag
        clock = task.Clock()
        d = task.deferLater(clock, 3, callable, 'foo', bar='bar')
        d.addCallback(self.assertIs, flag)
        clock.advance(2)
        self.assertEqual(results, [])
        clock.advance(1)
        self.assertEqual(results, [('foo', 'bar')])
        return d

    def test_errback(self):
        """
        The L{Deferred} returned by L{task.deferLater} is errbacked if the
        supplied function raises an exception.
        """

        def callable():
            raise TestException()
        clock = task.Clock()
        d = task.deferLater(clock, 1, callable)
        clock.advance(1)
        return self.assertFailure(d, TestException)

    def test_cancel(self):
        """
        The L{Deferred} returned by L{task.deferLater} can be
        cancelled to prevent the call from actually being performed.
        """
        called = []
        clock = task.Clock()
        d = task.deferLater(clock, 1, called.append, None)
        d.cancel()

        def cbCancelled(ignored):
            self.assertEqual([], clock.getDelayedCalls())
            self.assertFalse(called)
        self.assertFailure(d, defer.CancelledError)
        d.addCallback(cbCancelled)
        return d

    def test_noCallback(self):
        """
        The L{Deferred} returned by L{task.deferLater} fires with C{None}
        when no callback function is passed.
        """
        clock = task.Clock()
        d = task.deferLater(clock, 2.0)
        self.assertNoResult(d)
        clock.advance(2.0)
        self.assertIs(None, self.successResultOf(d))
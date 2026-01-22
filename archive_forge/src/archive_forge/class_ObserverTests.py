import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
class ObserverTests(unittest.SynchronousTestCase):
    """
    Tests for L{_synctest._LogObserver}, a helper for the implementation of
    L{SynchronousTestCase.flushLoggedErrors}.
    """

    def setUp(self):
        self.result = reporter.TestResult()
        self.observer = _synctest._LogObserver()

    def test_msg(self):
        """
        Test that a standard log message doesn't go anywhere near the result.
        """
        self.observer.gotEvent({'message': ('some message',), 'time': time.time(), 'isError': 0, 'system': '-'})
        self.assertEqual(self.observer.getErrors(), [])

    def test_error(self):
        """
        Test that an observed error gets added to the result
        """
        f = makeFailure()
        self.observer.gotEvent({'message': (), 'time': time.time(), 'isError': 1, 'system': '-', 'failure': f, 'why': None})
        self.assertEqual(self.observer.getErrors(), [f])

    def test_flush(self):
        """
        Check that flushing the observer with no args removes all errors.
        """
        self.test_error()
        flushed = self.observer.flushErrors()
        self.assertEqual(self.observer.getErrors(), [])
        self.assertEqual(len(flushed), 1)
        self.assertTrue(flushed[0].check(ZeroDivisionError))

    def _makeRuntimeFailure(self):
        return failure.Failure(RuntimeError('test error'))

    def test_flushByType(self):
        """
        Check that flushing the observer remove all failures of the given type.
        """
        self.test_error()
        f = self._makeRuntimeFailure()
        self.observer.gotEvent(dict(message=(), time=time.time(), isError=1, system='-', failure=f, why=None))
        flushed = self.observer.flushErrors(ZeroDivisionError)
        self.assertEqual(self.observer.getErrors(), [f])
        self.assertEqual(len(flushed), 1)
        self.assertTrue(flushed[0].check(ZeroDivisionError))

    def test_ignoreErrors(self):
        """
        Check that C{_ignoreErrors} actually causes errors to be ignored.
        """
        self.observer._ignoreErrors(ZeroDivisionError)
        f = makeFailure()
        self.observer.gotEvent({'message': (), 'time': time.time(), 'isError': 1, 'system': '-', 'failure': f, 'why': None})
        self.assertEqual(self.observer.getErrors(), [])

    def test_clearIgnores(self):
        """
        Check that C{_clearIgnores} ensures that previously ignored errors
        get captured.
        """
        self.observer._ignoreErrors(ZeroDivisionError)
        self.observer._clearIgnores()
        f = makeFailure()
        self.observer.gotEvent({'message': (), 'time': time.time(), 'isError': 1, 'system': '-', 'failure': f, 'why': None})
        self.assertEqual(self.observer.getErrors(), [f])
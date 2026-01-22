import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class ResultOfAssertionsTests(unittest.SynchronousTestCase):
    """
    Tests for L{SynchronousTestCase.successResultOf},
    L{SynchronousTestCase.failureResultOf}, and
    L{SynchronousTestCase.assertNoResult}.
    """
    result = object()
    failure = Failure(Exception('Bad times'))

    def test_withoutResult(self):
        """
        L{SynchronousTestCase.successResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with no current result.
        """
        self.assertRaises(self.failureException, self.successResultOf, Deferred())

    def test_successResultOfWithFailure(self):
        """
        L{SynchronousTestCase.successResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with a failure result.
        """
        self.assertRaises(self.failureException, self.successResultOf, fail(self.failure))

    def test_successResultOfWithFailureHasTraceback(self):
        """
        L{SynchronousTestCase.successResultOf} raises a
        L{SynchronousTestCase.failureException} that has the original failure
        traceback when called with a L{Deferred} with a failure result.
        """
        try:
            self.successResultOf(fail(self.failure))
        except self.failureException as e:
            self.assertIn(self.failure.getTraceback(), str(e))

    def test_failureResultOfWithoutResult(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with no current result.
        """
        self.assertRaises(self.failureException, self.failureResultOf, Deferred())

    def test_failureResultOfWithSuccess(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with a success result.
        """
        self.assertRaises(self.failureException, self.failureResultOf, succeed(self.result))

    def test_failureResultOfWithWrongFailure(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        that fails with an exception type that was not expected.
        """
        self.assertRaises(self.failureException, self.failureResultOf, fail(self.failure), KeyError)

    def test_failureResultOfWithWrongFailureOneExpectedFailure(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        that fails with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the expected
        exception type.
        """
        try:
            self.failureResultOf(fail(self.failure), KeyError)
        except self.failureException as e:
            self.assertIn('Failure of type ({}.{}) expected on'.format(KeyError.__module__, KeyError.__name__), str(e))

    def test_failureResultOfWithWrongFailureOneExpectedFailureHasTB(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        that fails with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the original
        failure traceback.
        """
        try:
            self.failureResultOf(fail(self.failure), KeyError)
        except self.failureException as e:
            self.assertIn(self.failure.getTraceback(), str(e))

    def test_failureResultOfWithWrongFailureMultiExpectedFailures(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the expected
        exception types in the error message.
        """
        try:
            self.failureResultOf(fail(self.failure), KeyError, IOError)
        except self.failureException as e:
            self.assertIn('Failure of type ({}.{} or {}.{}) expected on'.format(KeyError.__module__, KeyError.__name__, IOError.__module__, IOError.__name__), str(e))

    def test_failureResultOfWithWrongFailureMultiExpectedFailuresHasTB(self):
        """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a L{Deferred}
        with an exception type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the original
        failure traceback in the error message.
        """
        try:
            self.failureResultOf(fail(self.failure), KeyError, IOError)
        except self.failureException as e:
            self.assertIn(self.failure.getTraceback(), str(e))

    def test_successResultOfWithSuccessResult(self):
        """
        When passed a L{Deferred} which currently has a result (ie,
        L{Deferred.addCallback} would cause the added callback to be called
        before C{addCallback} returns), L{SynchronousTestCase.successResultOf}
        returns that result.
        """
        self.assertIdentical(self.result, self.successResultOf(succeed(self.result)))

    def test_failureResultOfWithExpectedFailureResult(self):
        """
        When passed a L{Deferred} which currently has a L{Failure} result (ie,
        L{Deferred.addErrback} would cause the added errback to be called
        before C{addErrback} returns), L{SynchronousTestCase.failureResultOf}
        returns that L{Failure} if its contained exception type is expected.
        """
        self.assertIdentical(self.failure, self.failureResultOf(fail(self.failure), self.failure.type, KeyError))

    def test_failureResultOfWithFailureResult(self):
        """
        When passed a L{Deferred} which currently has a L{Failure} result
        (ie, L{Deferred.addErrback} would cause the added errback to be called
        before C{addErrback} returns), L{SynchronousTestCase.failureResultOf}
        returns that L{Failure}.
        """
        self.assertIdentical(self.failure, self.failureResultOf(fail(self.failure)))

    def test_assertNoResultSuccess(self):
        """
        When passed a L{Deferred} which currently has a success result (see
        L{test_withSuccessResult}), L{SynchronousTestCase.assertNoResult}
        raises L{SynchronousTestCase.failureException}.
        """
        self.assertRaises(self.failureException, self.assertNoResult, succeed(self.result))

    def test_assertNoResultFailure(self):
        """
        When passed a L{Deferred} which currently has a failure result (see
        L{test_withFailureResult}), L{SynchronousTestCase.assertNoResult}
        raises L{SynchronousTestCase.failureException}.
        """
        self.assertRaises(self.failureException, self.assertNoResult, fail(self.failure))

    def test_assertNoResult(self):
        """
        When passed a L{Deferred} with no current result,
        L{SynchronousTestCase.assertNoResult} does not raise an exception.
        """
        self.assertNoResult(Deferred())

    def test_assertNoResultPropagatesSuccess(self):
        """
        When passed a L{Deferred} with no current result, which is then
        fired with a success result, L{SynchronousTestCase.assertNoResult}
        doesn't modify the result of the L{Deferred}.
        """
        d = Deferred()
        self.assertNoResult(d)
        d.callback(self.result)
        self.assertEqual(self.result, self.successResultOf(d))

    def test_assertNoResultPropagatesLaterFailure(self):
        """
        When passed a L{Deferred} with no current result, which is then
        fired with a L{Failure} result, L{SynchronousTestCase.assertNoResult}
        doesn't modify the result of the L{Deferred}.
        """
        d = Deferred()
        self.assertNoResult(d)
        d.errback(self.failure)
        self.assertEqual(self.failure, self.failureResultOf(d))

    def test_assertNoResultSwallowsImmediateFailure(self):
        """
        When passed a L{Deferred} which currently has a L{Failure} result,
        L{SynchronousTestCase.assertNoResult} changes the result of the
        L{Deferred} to a success.
        """
        d = fail(self.failure)
        try:
            self.assertNoResult(d)
        except self.failureException:
            pass
        self.assertEqual(None, self.successResultOf(d))
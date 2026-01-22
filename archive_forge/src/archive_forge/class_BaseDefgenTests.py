import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class BaseDefgenTests:
    """
    This class sets up a bunch of test cases which will test both
    deferredGenerator and inlineCallbacks based generators. The subclasses
    DeferredGeneratorTests and InlineCallbacksTests each provide the actual
    generator implementations tested.
    """

    def testBasics(self):
        """
        Test that a normal deferredGenerator works.  Tests yielding a
        deferred which callbacks, as well as a deferred errbacks. Also
        ensures returning a final value works.
        """
        return self._genBasics().addCallback(self.assertEqual, 'WOOSH')

    def testBuggy(self):
        """
        Ensure that a buggy generator properly signals a Failure
        condition on result deferred.
        """
        return self.assertFailure(self._genBuggy(), ZeroDivisionError)

    def testNothing(self):
        """Test that a generator which never yields results in None."""
        return self._genNothing().addCallback(self.assertEqual, None)

    def testHandledTerminalFailure(self):
        """
        Create a Deferred Generator which yields a Deferred which fails and
        handles the exception which results.  Assert that the Deferred
        Generator does not errback its Deferred.
        """
        return self._genHandledTerminalFailure().addCallback(self.assertEqual, None)

    def testHandledTerminalAsyncFailure(self):
        """
        Just like testHandledTerminalFailure, only with a Deferred which fires
        asynchronously with an error.
        """
        d = defer.Deferred()
        deferredGeneratorResultDeferred = self._genHandledTerminalAsyncFailure(d)
        d.errback(TerminalException('Handled Terminal Failure'))
        return deferredGeneratorResultDeferred.addCallback(self.assertEqual, None)

    def testStackUsage(self):
        """
        Make sure we don't blow the stack when yielding immediately
        available deferreds.
        """
        return self._genStackUsage().addCallback(self.assertEqual, 0)

    def testStackUsage2(self):
        """
        Make sure we don't blow the stack when yielding immediately
        available values.
        """
        return self._genStackUsage2().addCallback(self.assertEqual, 0)
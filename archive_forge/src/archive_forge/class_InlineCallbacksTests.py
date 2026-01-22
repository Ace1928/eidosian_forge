import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class InlineCallbacksTests(BaseDefgenTests, unittest.TestCase):

    def _genBasics(self):
        x = (yield getThing())
        self.assertEqual(x, 'hi')
        try:
            yield getOwie()
        except ZeroDivisionError as e:
            self.assertEqual(str(e), 'OMG')
        returnValue('WOOSH')
    _genBasics = inlineCallbacks(_genBasics)

    def _genBuggy(self):
        yield getThing()
        1 / 0
    _genBuggy = inlineCallbacks(_genBuggy)

    def _genNothing(self):
        if False:
            yield 1
    _genNothing = inlineCallbacks(_genNothing)

    def _genHandledTerminalFailure(self):
        try:
            yield defer.fail(TerminalException('Handled Terminal Failure'))
        except TerminalException:
            pass
    _genHandledTerminalFailure = inlineCallbacks(_genHandledTerminalFailure)

    def _genHandledTerminalAsyncFailure(self, d):
        try:
            yield d
        except TerminalException:
            pass
    _genHandledTerminalAsyncFailure = inlineCallbacks(_genHandledTerminalAsyncFailure)

    def _genStackUsage(self):
        for x in range(5000):
            yield defer.succeed(1)
        returnValue(0)
    _genStackUsage = inlineCallbacks(_genStackUsage)

    def _genStackUsage2(self):
        for x in range(5000):
            yield 1
        returnValue(0)
    _genStackUsage2 = inlineCallbacks(_genStackUsage2)

    def testYieldNonDeferred(self):
        """
        Ensure that yielding a non-deferred passes it back as the
        result of the yield expression.

        @return: A L{twisted.internet.defer.Deferred}
        @rtype: L{twisted.internet.defer.Deferred}
        """

        def _test():
            yield 5
            returnValue(5)
        _test = inlineCallbacks(_test)
        return _test().addCallback(self.assertEqual, 5)

    def testReturnNoValue(self):
        """Ensure a standard python return results in a None result."""

        def _noReturn():
            yield 5
            return
        _noReturn = inlineCallbacks(_noReturn)
        return _noReturn().addCallback(self.assertEqual, None)

    def testReturnValue(self):
        """Ensure that returnValue works."""

        def _return():
            yield 5
            returnValue(6)
        _return = inlineCallbacks(_return)
        return _return().addCallback(self.assertEqual, 6)

    def test_nonGeneratorReturn(self):
        """
        Ensure that C{TypeError} with a message about L{inlineCallbacks} is
        raised when a non-generator returns something other than a generator.
        """

        def _noYield():
            return 5
        _noYield = inlineCallbacks(_noYield)
        self.assertIn('inlineCallbacks', str(self.assertRaises(TypeError, _noYield)))

    def test_nonGeneratorReturnValue(self):
        """
        Ensure that C{TypeError} with a message about L{inlineCallbacks} is
        raised when a non-generator calls L{returnValue}.
        """

        def _noYield():
            returnValue(5)
        _noYield = inlineCallbacks(_noYield)
        self.assertIn('inlineCallbacks', str(self.assertRaises(TypeError, _noYield)))

    def test_internalDefGenReturnValueDoesntLeak(self):
        """
        When one inlineCallbacks calls another, the internal L{_DefGen_Return}
        flow control exception raised by calling L{defer.returnValue} doesn't
        leak into tracebacks captured in the caller.
        """
        clock = task.Clock()

        @inlineCallbacks
        def _returns():
            """
            This is the inner function using returnValue.
            """
            yield task.deferLater(clock, 0)
            returnValue('actual-value-not-used-for-the-test')

        @inlineCallbacks
        def _raises():
            try:
                yield _returns()
                raise TerminalException('boom returnValue')
            except TerminalException:
                return traceback.format_exc()
        d = _raises()
        clock.advance(0)
        tb = self.successResultOf(d)
        self.assertNotIn('_DefGen_Return', tb)
        self.assertNotIn('During handling of the above exception, another exception occurred', tb)
        self.assertIn('test_defgen.TerminalException: boom returnValue', tb)

    def test_internalStopIterationDoesntLeak(self):
        """
        When one inlineCallbacks calls another, the internal L{StopIteration}
        flow control exception generated when the inner generator returns
        doesn't leak into tracebacks captured in the caller.

        This is similar to C{test_internalDefGenReturnValueDoesntLeak} but the
        inner function uses the "normal" return statemement rather than the
        C{returnValue} helper.
        """
        clock = task.Clock()

        @inlineCallbacks
        def _returns():
            yield task.deferLater(clock, 0)
            return 6

        @inlineCallbacks
        def _raises():
            try:
                yield _returns()
                raise TerminalException('boom normal return')
            except TerminalException:
                return traceback.format_exc()
        d = _raises()
        clock.advance(0)
        tb = self.successResultOf(d)
        self.assertNotIn('StopIteration', tb)
        self.assertNotIn('During handling of the above exception, another exception occurred', tb)
        self.assertIn('test_defgen.TerminalException: boom normal return', tb)
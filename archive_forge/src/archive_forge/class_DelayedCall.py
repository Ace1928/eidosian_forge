from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class DelayedCall(unittest.TestCase):
    hiddenExceptionMsg = 'something blew up'

    def go(self):
        raise RuntimeError(self.hiddenExceptionMsg)

    def testHiddenException(self):
        """
        What happens if an error is raised in a DelayedCall and an error is
        also raised in the test?

        L{test_reporter.ErrorReportingTests.testHiddenException} checks that
        both errors get reported.

        Note that this behaviour is deprecated. A B{real} test would return a
        Deferred that got triggered by the callLater. This would guarantee the
        delayed call error gets reported.
        """
        reactor.callLater(0, self.go)
        reactor.iterate(0.01)
        self.fail('Deliberate failure to mask the hidden exception')
    testHiddenException.suppress = [util.suppress(message='reactor\\.iterate cannot be used.*', category=DeprecationWarning)]
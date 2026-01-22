from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def _cancelConnectTest(self, connect):
    """
        Helper for implementing a test to verify that cancellation of the
        L{Deferred} returned by one of L{ClientCreator}'s I{connect} methods is
        implemented to cancel the underlying connector.

        @param connect: A function which will be invoked with a L{ClientCreator}
            instance as an argument and which should call one its I{connect}
            methods and return the result.

        @return: A L{Deferred} which fires when the test is complete or fails if
            there is a problem.
        """
    reactor = MemoryReactorClock()
    cc = ClientCreator(reactor, Protocol)
    d = connect(cc)
    connector = reactor.connectors.pop()
    self.assertFalse(connector._disconnected)
    d.cancel()
    self.assertTrue(connector._disconnected)
    return self.assertFailure(d, CancelledError)
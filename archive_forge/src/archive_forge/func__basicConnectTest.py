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
def _basicConnectTest(self, check):
    """
        Helper for implementing a test to verify that one of the I{connect}
        methods of L{ClientCreator} passes the right arguments to the right
        reactor method.

        @param check: A function which will be invoked with a reactor and a
            L{ClientCreator} instance and which should call one of the
            L{ClientCreator}'s I{connect} methods and assert that all of its
            arguments except for the factory are passed on as expected to the
            reactor.  The factory should be returned.
        """

    class SomeProtocol(Protocol):
        pass
    reactor = MemoryReactorClock()
    cc = ClientCreator(reactor, SomeProtocol)
    factory = check(reactor, cc)
    protocol = factory.buildProtocol(None)
    self.assertIsInstance(protocol, SomeProtocol)
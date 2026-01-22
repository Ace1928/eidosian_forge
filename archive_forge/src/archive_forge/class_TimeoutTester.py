import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class TimeoutTester(protocol.Protocol, policies.TimeoutMixin):
    """
    A testable protocol with timeout facility.

    @ivar timedOut: set to C{True} if a timeout has been detected.
    @type timedOut: C{bool}
    """
    timeOut = 3
    timedOut = False

    def __init__(self, clock):
        """
        Initialize the protocol with a C{task.Clock} object.
        """
        self.clock = clock

    def connectionMade(self):
        """
        Upon connection, set the timeout.
        """
        self.setTimeout(self.timeOut)

    def dataReceived(self, data):
        """
        Reset the timeout on data.
        """
        self.resetTimeout()
        protocol.Protocol.dataReceived(self, data)

    def connectionLost(self, reason=None):
        """
        On connection lost, cancel all timeout operations.
        """
        self.setTimeout(None)

    def timeoutConnection(self):
        """
        Flags the timedOut variable to indicate the timeout of the connection.
        """
        self.timedOut = True

    def callLater(self, timeout, func, *args, **kwargs):
        """
        Override callLater to use the deterministic clock.
        """
        return self.clock.callLater(timeout, func, *args, **kwargs)
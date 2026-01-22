import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class TimeoutProtocol(ProtocolWrapper):
    """
    Protocol that automatically disconnects when the connection is idle.
    """

    def __init__(self, factory, wrappedProtocol, timeoutPeriod):
        """
        Constructor.

        @param factory: An L{TimeoutFactory}.
        @param wrappedProtocol: A L{Protocol} to wrapp.
        @param timeoutPeriod: Number of seconds to wait for activity before
            timing out.
        """
        ProtocolWrapper.__init__(self, factory, wrappedProtocol)
        self.timeoutCall = None
        self.timeoutPeriod = None
        self.setTimeout(timeoutPeriod)

    def setTimeout(self, timeoutPeriod=None):
        """
        Set a timeout.

        This will cancel any existing timeouts.

        @param timeoutPeriod: If not L{None}, change the timeout period.
            Otherwise, use the existing value.
        """
        self.cancelTimeout()
        self.timeoutPeriod = timeoutPeriod
        if timeoutPeriod is not None:
            self.timeoutCall = self.factory.callLater(self.timeoutPeriod, self.timeoutFunc)

    def cancelTimeout(self):
        """
        Cancel the timeout.

        If the timeout was already cancelled, this does nothing.
        """
        self.timeoutPeriod = None
        if self.timeoutCall:
            try:
                self.timeoutCall.cancel()
            except (error.AlreadyCalled, error.AlreadyCancelled):
                pass
            self.timeoutCall = None

    def resetTimeout(self):
        """
        Reset the timeout, usually because some activity just happened.
        """
        if self.timeoutCall:
            self.timeoutCall.reset(self.timeoutPeriod)

    def write(self, data):
        self.resetTimeout()
        ProtocolWrapper.write(self, data)

    def writeSequence(self, seq):
        self.resetTimeout()
        ProtocolWrapper.writeSequence(self, seq)

    def dataReceived(self, data):
        self.resetTimeout()
        ProtocolWrapper.dataReceived(self, data)

    def connectionLost(self, reason):
        self.cancelTimeout()
        ProtocolWrapper.connectionLost(self, reason)

    def timeoutFunc(self):
        """
        This method is called when the timeout is triggered.

        By default it calls I{loseConnection}.  Override this if you want
        something else to happen.
        """
        self.loseConnection()
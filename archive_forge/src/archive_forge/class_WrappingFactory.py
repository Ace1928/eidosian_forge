import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class WrappingFactory(ClientFactory):
    """
    Wraps a factory and its protocols, and keeps track of them.
    """
    protocol: Type[Protocol] = ProtocolWrapper

    def __init__(self, wrappedFactory):
        self.wrappedFactory = wrappedFactory
        self.protocols = {}

    def logPrefix(self):
        """
        Generate a log prefix mentioning both the wrapped factory and this one.
        """
        return _wrappedLogPrefix(self, self.wrappedFactory)

    def doStart(self):
        self.wrappedFactory.doStart()
        ClientFactory.doStart(self)

    def doStop(self):
        self.wrappedFactory.doStop()
        ClientFactory.doStop(self)

    def startedConnecting(self, connector):
        self.wrappedFactory.startedConnecting(connector)

    def clientConnectionFailed(self, connector, reason):
        self.wrappedFactory.clientConnectionFailed(connector, reason)

    def clientConnectionLost(self, connector, reason):
        self.wrappedFactory.clientConnectionLost(connector, reason)

    def buildProtocol(self, addr):
        return self.protocol(self, self.wrappedFactory.buildProtocol(addr))

    def registerProtocol(self, p):
        """
        Called by protocol to register itself.
        """
        self.protocols[p] = 1

    def unregisterProtocol(self, p):
        """
        Called by protocols when they go away.
        """
        del self.protocols[p]
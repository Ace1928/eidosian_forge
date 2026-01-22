import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class ProtocolWrapper(Protocol):
    """
    Wraps protocol instances and acts as their transport as well.

    @ivar wrappedProtocol: An L{IProtocol<twisted.internet.interfaces.IProtocol>}
        provider to which L{IProtocol<twisted.internet.interfaces.IProtocol>}
        method calls onto this L{ProtocolWrapper} will be proxied.

    @ivar factory: The L{WrappingFactory} which created this
        L{ProtocolWrapper}.
    """
    disconnecting = 0

    def __init__(self, factory: 'WrappingFactory', wrappedProtocol: interfaces.IProtocol):
        self.wrappedProtocol = wrappedProtocol
        self.factory = factory

    def logPrefix(self):
        """
        Use a customized log prefix mentioning both the wrapped protocol and
        the current one.
        """
        return _wrappedLogPrefix(self, self.wrappedProtocol)

    def makeConnection(self, transport):
        """
        When a connection is made, register this wrapper with its factory,
        save the real transport, and connect the wrapped protocol to this
        L{ProtocolWrapper} to intercept any transport calls it makes.
        """
        directlyProvides(self, providedBy(transport))
        Protocol.makeConnection(self, transport)
        self.factory.registerProtocol(self)
        self.wrappedProtocol.makeConnection(self)

    def write(self, data):
        self.transport.write(data)

    def writeSequence(self, data):
        self.transport.writeSequence(data)

    def loseConnection(self):
        self.disconnecting = 1
        self.transport.loseConnection()

    def getPeer(self):
        return self.transport.getPeer()

    def getHost(self):
        return self.transport.getHost()

    def registerProducer(self, producer, streaming):
        self.transport.registerProducer(producer, streaming)

    def unregisterProducer(self):
        self.transport.unregisterProducer()

    def stopConsuming(self):
        self.transport.stopConsuming()

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def dataReceived(self, data):
        self.wrappedProtocol.dataReceived(data)

    def connectionLost(self, reason):
        self.factory.unregisterProtocol(self)
        self.wrappedProtocol.connectionLost(reason)
        self.wrappedProtocol = None
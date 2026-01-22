import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class ThrottlingProtocol(ProtocolWrapper):
    """
    Protocol for L{ThrottlingFactory}.
    """

    def write(self, data):
        self.factory.registerWritten(len(data))
        ProtocolWrapper.write(self, data)

    def writeSequence(self, seq):
        self.factory.registerWritten(sum(map(len, seq)))
        ProtocolWrapper.writeSequence(self, seq)

    def dataReceived(self, data):
        self.factory.registerRead(len(data))
        ProtocolWrapper.dataReceived(self, data)

    def registerProducer(self, producer, streaming):
        self.producer = producer
        ProtocolWrapper.registerProducer(self, producer, streaming)

    def unregisterProducer(self):
        del self.producer
        ProtocolWrapper.unregisterProducer(self)

    def throttleReads(self):
        self.transport.pauseProducing()

    def unthrottleReads(self):
        self.transport.resumeProducing()

    def throttleWrites(self):
        if hasattr(self, 'producer'):
            self.producer.pauseProducing()

    def unthrottleWrites(self):
        if hasattr(self, 'producer'):
            self.producer.resumeProducing()
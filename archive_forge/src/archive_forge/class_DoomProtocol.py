from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class DoomProtocol(SimpleProtocol):
    i = 0

    def lineReceived(self, line):
        self.i += 1
        if self.i < 4:
            self.sendLine(b'Hello %d' % (self.i,))
        SimpleProtocol.lineReceived(self, line)
        if self.lines[-1] == b'Hello 3':
            self.transport.loseConnection()
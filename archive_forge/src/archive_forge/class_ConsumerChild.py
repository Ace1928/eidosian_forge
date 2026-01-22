import sys
from twisted.internet import protocol, stdio
from twisted.protocols import basic
from twisted.python import log, reflect
class ConsumerChild(protocol.Protocol):

    def __init__(self, junkPath):
        self.junkPath = junkPath

    def connectionMade(self):
        d = basic.FileSender().beginFileTransfer(open(self.junkPath, 'rb'), self.transport)
        d.addErrback(failed)
        d.addCallback(lambda ign: self.transport.loseConnection())

    def connectionLost(self, reason):
        reactor.stop()
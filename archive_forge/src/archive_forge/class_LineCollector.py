import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
class LineCollector(basic.LineReceiver):
    """
    @ivar deferred: a deferred that will fire at connection lost.
    @type deferred: L{defer.Deferred}

    @ivar doTLS: whether the protocol is initiate TLS or not.
    @type doTLS: C{bool}

    @ivar fillBuffer: if set to True, it will send lots of data once
        C{STARTTLS} is received.
    @type fillBuffer: C{bool}
    """

    def __init__(self, doTLS, fillBuffer=False):
        self.doTLS = doTLS
        self.fillBuffer = fillBuffer
        self.deferred = defer.Deferred()

    def connectionMade(self):
        self.factory.rawdata = b''
        self.factory.lines = []

    def lineReceived(self, line):
        self.factory.lines.append(line)
        if line == b'STARTTLS':
            if self.fillBuffer:
                for x in range(500):
                    self.sendLine(b'X' * 1000)
            self.sendLine(b'READY')
            if self.doTLS:
                ctx = ServerTLSContext(privateKeyFileName=certPath, certificateFileName=certPath)
                self.transport.startTLS(ctx, self.factory.server)
            else:
                self.setRawMode()

    def rawDataReceived(self, data):
        self.factory.rawdata += data
        self.transport.loseConnection()

    def connectionLost(self, reason):
        self.deferred.callback(None)
import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class TrafficLoggingProtocol(ProtocolWrapper):

    def __init__(self, factory, wrappedProtocol, logfile, lengthLimit=None, number=0):
        """
        @param factory: factory which created this protocol.
        @type factory: L{protocol.Factory}.
        @param wrappedProtocol: the underlying protocol.
        @type wrappedProtocol: C{protocol.Protocol}.
        @param logfile: file opened for writing used to write log messages.
        @type logfile: C{file}
        @param lengthLimit: maximum size of the datareceived logged.
        @type lengthLimit: C{int}
        @param number: identifier of the connection.
        @type number: C{int}.
        """
        ProtocolWrapper.__init__(self, factory, wrappedProtocol)
        self.logfile = logfile
        self.lengthLimit = lengthLimit
        self._number = number

    def _log(self, line):
        self.logfile.write(line + '\n')
        self.logfile.flush()

    def _mungeData(self, data):
        if self.lengthLimit and len(data) > self.lengthLimit:
            data = data[:self.lengthLimit - 12] + '<... elided>'
        return data

    def connectionMade(self):
        self._log('*')
        return ProtocolWrapper.connectionMade(self)

    def dataReceived(self, data):
        self._log('C %d: %r' % (self._number, self._mungeData(data)))
        return ProtocolWrapper.dataReceived(self, data)

    def connectionLost(self, reason):
        self._log('C %d: %r' % (self._number, reason))
        return ProtocolWrapper.connectionLost(self, reason)

    def write(self, data):
        self._log('S %d: %r' % (self._number, self._mungeData(data)))
        return ProtocolWrapper.write(self, data)

    def writeSequence(self, iovec):
        self._log('SV %d: %r' % (self._number, [self._mungeData(d) for d in iovec]))
        return ProtocolWrapper.writeSequence(self, iovec)

    def loseConnection(self):
        self._log('S %d: *' % (self._number,))
        return ProtocolWrapper.loseConnection(self)
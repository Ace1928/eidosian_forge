import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
@implementer(interfaces.ITransport)
class SSHSessionProcessProtocol(protocol.ProcessProtocol):
    """I am both an L{IProcessProtocol} and an L{ITransport}.

    I am a transport to the remote endpoint and a process protocol to the
    local subsystem.
    """
    _signalValuesToNames = None

    def __init__(self, session):
        self.session = session
        self.lostOutOrErrFlag = False

    def connectionMade(self):
        if self.session.buf:
            self.transport.write(self.session.buf)
            self.session.buf = None

    def outReceived(self, data):
        self.session.write(data)

    def errReceived(self, err):
        self.session.writeExtended(connection.EXTENDED_DATA_STDERR, err)

    def outConnectionLost(self):
        """
        EOF should only be sent when both STDOUT and STDERR have been closed.
        """
        if self.lostOutOrErrFlag:
            self.session.conn.sendEOF(self.session)
        else:
            self.lostOutOrErrFlag = True

    def errConnectionLost(self):
        """
        See outConnectionLost().
        """
        self.outConnectionLost()

    def connectionLost(self, reason=None):
        self.session.loseConnection()

    def _getSignalName(self, signum):
        """
        Get a signal name given a signal number.
        """
        if self._signalValuesToNames is None:
            self._signalValuesToNames = {}
            for signame in SUPPORTED_SIGNALS:
                signame = 'SIG' + signame
                sigvalue = getattr(signal, signame, None)
                if sigvalue is not None:
                    self._signalValuesToNames[sigvalue] = signame
            for k, v in signal.__dict__.items():
                if k.startswith('SIG') and (not k.startswith('SIG_')):
                    if v not in self._signalValuesToNames:
                        self._signalValuesToNames[v] = k + '@' + sys.platform
        return self._signalValuesToNames[signum]

    def processEnded(self, reason=None):
        """
        When we are told the process ended, try to notify the other side about
        how the process ended using the exit-signal or exit-status requests.
        Also, close the channel.
        """
        if reason is not None:
            err = reason.value
            if err.signal is not None:
                signame = self._getSignalName(err.signal)
                if getattr(os, 'WCOREDUMP', None) is not None and os.WCOREDUMP(err.status):
                    log.info('exitSignal: {signame} (core dumped)', signame=signame)
                    coreDumped = True
                else:
                    log.info('exitSignal: {}', signame=signame)
                    coreDumped = False
                self.session.conn.sendRequest(self.session, b'exit-signal', common.NS(networkString(signame[3:])) + (b'\x01' if coreDumped else b'\x00') + common.NS(b'') + common.NS(b''))
            elif err.exitCode is not None:
                log.info('exitCode: {exitCode!r}', exitCode=err.exitCode)
                self.session.conn.sendRequest(self.session, b'exit-status', struct.pack('>L', err.exitCode))
        self.session.loseConnection()

    def getHost(self):
        """
        Return the host from my session's transport.
        """
        return self.session.conn.transport.getHost()

    def getPeer(self):
        """
        Return the peer from my session's transport.
        """
        return self.session.conn.transport.getPeer()

    def write(self, data):
        self.session.write(data)

    def writeSequence(self, seq):
        self.session.write(b''.join(seq))

    def loseConnection(self):
        self.session.loseConnection()
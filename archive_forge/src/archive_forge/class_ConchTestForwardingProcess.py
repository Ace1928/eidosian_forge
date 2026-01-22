import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import (
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
import sys, os
from twisted.conch.scripts.%s import run
class ConchTestForwardingProcess(protocol.ProcessProtocol):
    """
    Manages a third-party process which launches a server.

    Uses L{ConchTestForwardingPort} to connect to the third-party server.
    Once L{ConchTestForwardingPort} has disconnected, kill the process and fire
    a Deferred with the data received by the L{ConchTestForwardingPort}.

    @ivar deferred: Set by whatever uses this object. Accessed using
    L{_getDeferred}, which destroys the value so the Deferred is not
    fired twice. Fires when the process is terminated.
    """
    deferred = None

    def __init__(self, port, data):
        """
        @type port: L{int}
        @param port: The port on which the third-party server is listening.
        (it is assumed that the server is running on localhost).

        @type data: L{str}
        @param data: This is sent to the third-party server. Must end with '
'
        in order to trigger a disconnect.
        """
        self.port = port
        self.buffer = None
        self.data = data

    def _getDeferred(self):
        d, self.deferred = (self.deferred, None)
        return d

    def connectionMade(self):
        self._connect()

    def _connect(self):
        """
        Connect to the server, which is often a third-party process.
        Tries to reconnect if it fails because we have no way of determining
        exactly when the port becomes available for listening -- we can only
        know when the process starts.
        """
        cc = protocol.ClientCreator(reactor, ConchTestForwardingPort, self, self.data)
        d = cc.connectTCP('127.0.0.1', self.port)
        d.addErrback(self._ebConnect)
        return d

    def _ebConnect(self, f):
        reactor.callLater(0.1, self._connect)

    def forwardingPortDisconnected(self, buffer):
        """
        The network connection has died; save the buffer of output
        from the network and attempt to quit the process gracefully,
        and then (after the reactor has spun) send it a KILL signal.
        """
        self.buffer = buffer
        self.transport.write(b'\x03')
        self.transport.loseConnection()
        reactor.callLater(0, self._reallyDie)

    def _reallyDie(self):
        try:
            self.transport.signalProcess('KILL')
        except ProcessExitedAlready:
            pass

    def processEnded(self, reason):
        """
        Fire the Deferred at self.deferred with the data collected
        from the L{ConchTestForwardingPort} connection, if any.
        """
        self._getDeferred().callback(self.buffer)
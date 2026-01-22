import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
class StubTransport:
    """
    A stub transport which records the data written.

    @ivar buf: the data sent to the transport.
    @type buf: L{bytes}

    @ivar close: flags indicating if the transport has been closed.
    @type close: L{bool}
    """
    buf = b''
    close = False

    def getPeer(self):
        """
        Return an arbitrary L{IAddress}.
        """
        return IPv4Address('TCP', 'remotehost', 8888)

    def getHost(self):
        """
        Return an arbitrary L{IAddress}.
        """
        return IPv4Address('TCP', 'localhost', 9999)

    def write(self, data):
        """
        Record data in the buffer.
        """
        self.buf += data

    def loseConnection(self):
        """
        Note that the connection was closed.
        """
        self.close = True

    def setTcpNoDelay(self, enabled):
        """
        Pretend to set C{TCP_NODELAY}.
        """
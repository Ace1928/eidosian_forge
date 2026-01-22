import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def _reprTest(self, serverProto, protocolName):
    """
        Test the C{__str__} and C{__repr__} implementations of a UNIX datagram
        port when used with the given protocol.
        """
    filename = self.mktemp()
    unixPort = reactor.listenUNIXDatagram(filename, serverProto)
    connectedString = f'<{protocolName} on {filename!r}>'
    self.assertEqual(repr(unixPort), connectedString)
    self.assertEqual(str(unixPort), connectedString)
    stopDeferred = defer.maybeDeferred(unixPort.stopListening)

    def stoppedListening(ign):
        unconnectedString = f'<{protocolName} (not listening)>'
        self.assertEqual(repr(unixPort), unconnectedString)
        self.assertEqual(str(unixPort), unconnectedString)
    stopDeferred.addCallback(stoppedListening)
    return stopDeferred
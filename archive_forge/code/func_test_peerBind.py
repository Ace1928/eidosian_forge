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
def test_peerBind(self):
    """
        The address passed to the server factory's C{buildProtocol} method and
        the address returned by the connected protocol's transport's C{getPeer}
        method match the address the client socket is bound to.
        """
    filename = self.mktemp()
    peername = self.mktemp()
    serverFactory = MyServerFactory()
    connMade = serverFactory.protocolConnectionMade = defer.Deferred()
    unixPort = reactor.listenUNIX(filename, serverFactory)
    self.addCleanup(unixPort.stopListening)
    unixSocket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.addCleanup(unixSocket.close)
    unixSocket.bind(peername)
    unixSocket.connect(filename)

    def cbConnMade(proto):
        expected = address.UNIXAddress(peername)
        self.assertEqual(serverFactory.peerAddresses, [expected])
        self.assertEqual(proto.transport.getPeer(), expected)
    connMade.addCallback(cbConnMade)
    return connMade
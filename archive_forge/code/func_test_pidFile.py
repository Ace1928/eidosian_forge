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
def test_pidFile(self):
    """
        A lockfile is created and locked when L{IReactorUNIX.listenUNIX} is
        called and released when the Deferred returned by the L{IListeningPort}
        provider's C{stopListening} method is called back.
        """
    filename = self.mktemp()
    serverFactory = MyServerFactory()
    serverConnMade = defer.Deferred()
    serverFactory.protocolConnectionMade = serverConnMade
    unixPort = reactor.listenUNIX(filename, serverFactory, wantPID=True)
    self.assertTrue(lockfile.isLocked(filename + '.lock'))
    clientFactory = MyClientFactory()
    clientConnMade = defer.Deferred()
    clientFactory.protocolConnectionMade = clientConnMade
    reactor.connectUNIX(filename, clientFactory, checkPID=1)
    d = defer.gatherResults([serverConnMade, clientConnMade])

    def _portStuff(args):
        serverProtocol, clientProto = args
        self.assertEqual(clientFactory.peerAddresses, [address.UNIXAddress(filename)])
        clientProto.transport.loseConnection()
        serverProtocol.transport.loseConnection()
        return unixPort.stopListening()
    d.addCallback(_portStuff)

    def _check(ignored):
        self.assertFalse(lockfile.isLocked(filename + '.lock'), 'locked')
    d.addCallback(_check)
    return d
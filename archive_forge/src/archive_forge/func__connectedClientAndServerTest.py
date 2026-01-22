import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def _connectedClientAndServerTest(self, callback):
    """
        Invoke the given callback with a client protocol and a server protocol
        which have been connected to each other.
        """
    serverFactory = MyServerFactory()
    serverConnMade = defer.Deferred()
    serverFactory.protocolConnectionMade = serverConnMade
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    portNumber = port.getHost().port
    clientF = MyClientFactory()
    clientConnMade = defer.Deferred()
    clientF.protocolConnectionMade = clientConnMade
    reactor.connectTCP('127.0.0.1', portNumber, clientF)
    connsMade = defer.gatherResults([serverConnMade, clientConnMade])

    def connected(result):
        serverProtocol, clientProtocol = result
        callback(serverProtocol, clientProtocol)
        serverProtocol.transport.loseConnection()
        clientProtocol.transport.loseConnection()
    connsMade.addCallback(connected)
    return connsMade
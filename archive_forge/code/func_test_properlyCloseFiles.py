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
def test_properlyCloseFiles(self):
    """
        Test that lost connections properly have their underlying socket
        resources cleaned up.
        """
    onServerConnectionLost = defer.Deferred()
    serverFactory = protocol.ServerFactory()
    serverFactory.protocol = lambda: ConnectionLostNotifyingProtocol(onServerConnectionLost)
    serverPort = self.createServer('127.0.0.1', 0, serverFactory)
    onClientConnectionLost = defer.Deferred()
    serverAddr = serverPort.getHost()
    clientCreator = protocol.ClientCreator(reactor, lambda: HandleSavingProtocol(onClientConnectionLost))
    clientDeferred = self.connectClient(serverAddr.host, serverAddr.port, clientCreator)

    def clientConnected(client):
        """
            Disconnect the client.  Return a Deferred which fires when both
            the client and the server have received disconnect notification.
            """
        client.transport.write(b'some bytes to make sure the connection is set up')
        client.transport.loseConnection()
        return defer.gatherResults([onClientConnectionLost, onServerConnectionLost])
    clientDeferred.addCallback(clientConnected)

    def clientDisconnected(result):
        """
            Verify that the underlying platform socket handle has been
            cleaned up.
            """
        client, server = result
        if not client.lostConnectionReason.check(error.ConnectionClosed):
            err(client.lostConnectionReason, 'Client lost connection for unexpected reason')
        if not server.lostConnectionReason.check(error.ConnectionClosed):
            err(server.lostConnectionReason, 'Server lost connection for unexpected reason')
        errorCodeMatcher = self.getHandleErrorCodeMatcher()
        exception = self.assertRaises(self.getHandleExceptionType(), client.handle.send, b'bytes')
        hamcrest.assert_that(exception.args[0], errorCodeMatcher)
    clientDeferred.addCallback(clientDisconnected)

    def cleanup(passthrough):
        """
            Shut down the server port.  Return a Deferred which fires when
            this has completed.
            """
        result = defer.maybeDeferred(serverPort.stopListening)
        result.addCallback(lambda ign: passthrough)
        return result
    clientDeferred.addBoth(cleanup)
    return clientDeferred
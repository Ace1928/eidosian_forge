from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_writeAfterHandshake(self):
    """
        Bytes written to L{TLSMemoryBIOProtocol} before the handshake is
        complete are received by the protocol on the other side of the
        connection once the handshake succeeds.
        """
    data = b'some bytes'
    clientProtocol = Protocol()
    clientFactory = ClientFactory()
    clientFactory.protocol = lambda: clientProtocol
    clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
    wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
    sslClientProtocol = wrapperFactory.buildProtocol(None)
    serverProtocol = AccumulatingProtocol(len(data))
    serverFactory = ServerFactory()
    serverFactory.protocol = lambda: serverProtocol
    serverContextFactory = ServerTLSContext()
    wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
    sslServerProtocol = wrapperFactory.buildProtocol(None)
    connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

    def cbHandshook(ignored):
        clientProtocol.transport.write(data)
        return connectionDeferred
    handshakeDeferred.addCallback(cbHandshook)

    def cbDisconnected(ignored):
        self.assertEqual(b''.join(serverProtocol.received), data)
    handshakeDeferred.addCallback(cbDisconnected)
    return handshakeDeferred
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
def writeBeforeHandshakeTest(self, sendingProtocol, data):
    """
        Run test where client sends data before handshake, given the sending
        protocol and expected bytes.
        """
    clientFactory = ClientFactory()
    clientFactory.protocol = sendingProtocol
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

    def cbConnectionDone(ignored):
        self.assertEqual(b''.join(serverProtocol.received), data)
    connectionDeferred.addCallback(cbConnectionDone)
    return connectionDeferred
import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def _loopbackTLSConnection(serverOpts, clientOpts):
    """
    Common implementation code for both L{loopbackTLSConnection} and
    L{loopbackTLSConnectionInMemory}. Creates a loopback TLS connection
    using the provided server and client context factories.

    @param serverOpts: An OpenSSL context factory for the server.
    @type serverOpts: C{OpenSSLCertificateOptions}, or any class with an
        equivalent API.

    @param clientOpts: An OpenSSL context factory for the client.
    @type clientOpts: C{OpenSSLCertificateOptions}, or any class with an
        equivalent API.

    @return: 5-tuple of server-tls-protocol, server-inner-protocol,
        client-tls-protocol, client-inner-protocol and L{IOPump}
    @rtype: L{tuple}
    """

    class GreetingServer(protocol.Protocol):
        greeting = b'greetings!'

        def connectionMade(self):
            self.transport.write(self.greeting)

    class ListeningClient(protocol.Protocol):
        data = b''
        lostReason = None

        def dataReceived(self, data):
            self.data += data

        def connectionLost(self, reason):
            self.lostReason = reason
    clientWrappedProto = ListeningClient()
    serverWrappedProto = GreetingServer()
    plainClientFactory = protocol.Factory()
    plainClientFactory.protocol = lambda: clientWrappedProto
    plainServerFactory = protocol.Factory()
    plainServerFactory.protocol = lambda: serverWrappedProto
    clock = Clock()
    clientFactory = TLSMemoryBIOFactory(clientOpts, isClient=True, wrappedFactory=plainServerFactory, clock=clock)
    serverFactory = TLSMemoryBIOFactory(serverOpts, isClient=False, wrappedFactory=plainClientFactory, clock=clock)
    sProto, cProto, pump = connectedServerAndClient(lambda: serverFactory.buildProtocol(None), lambda: clientFactory.buildProtocol(None), clock=clock)
    pump.flush()
    return (sProto, cProto, serverWrappedProto, clientWrappedProto, pump)
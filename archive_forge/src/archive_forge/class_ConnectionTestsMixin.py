import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
class ConnectionTestsMixin:
    """
    This mixin defines test methods which should apply to most L{ITransport}
    implementations.
    """
    endpoints: Optional[EndpointCreator] = None

    def test_logPrefix(self):
        """
        Client and server transports implement L{ILoggingContext.logPrefix} to
        return a message reflecting the protocol they are running.
        """

        class CustomLogPrefixProtocol(ConnectableProtocol):

            def __init__(self, prefix):
                self._prefix = prefix
                self.system = None

            def connectionMade(self):
                self.transport.write(b'a')

            def logPrefix(self):
                return self._prefix

            def dataReceived(self, bytes):
                self.system = context.get(ILogContext)['system']
                self.transport.write(b'b')
                if b'b' in bytes:
                    self.transport.loseConnection()
        client = CustomLogPrefixProtocol('Custom Client')
        server = CustomLogPrefixProtocol('Custom Server')
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertIn('Custom Client', client.system)
        self.assertIn('Custom Server', server.system)

    def test_writeAfterDisconnect(self):
        """
        After a connection is disconnected, L{ITransport.write} and
        L{ITransport.writeSequence} are no-ops.
        """
        reactor = self.buildReactor()
        finished = []
        serverConnectionLostDeferred = Deferred()
        protocol = lambda: ClosingLaterProtocol(serverConnectionLostDeferred)
        portDeferred = self.endpoints.server(reactor).listen(ServerFactory.forProtocol(protocol))

        def listening(port):
            msg(f'Listening on {port.getHost()!r}')
            endpoint = self.endpoints.client(reactor, port.getHost())
            lostConnectionDeferred = Deferred()
            protocol = lambda: ClosingLaterProtocol(lostConnectionDeferred)
            client = endpoint.connect(ClientFactory.forProtocol(protocol))

            def write(proto):
                msg(f'About to write to {proto!r}')
                proto.transport.write(b'x')
            client.addCallbacks(write, lostConnectionDeferred.errback)

            def disconnected(proto):
                msg(f'{proto!r} disconnected')
                proto.transport.write(b'some bytes to get lost')
                proto.transport.writeSequence([b'some', b'more'])
                finished.append(True)
            lostConnectionDeferred.addCallback(disconnected)
            serverConnectionLostDeferred.addCallback(disconnected)
            return gatherResults([lostConnectionDeferred, serverConnectionLostDeferred])

        def onListen():
            portDeferred.addCallback(listening)
            portDeferred.addErrback(err)
            portDeferred.addCallback(lambda ignored: reactor.stop())
        needsRunningReactor(reactor, onListen)
        self.runReactor(reactor)
        self.assertEqual(finished, [True, True])

    def test_protocolGarbageAfterLostConnection(self):
        """
        After the connection a protocol is being used for is closed, the
        reactor discards all of its references to the protocol.
        """
        lostConnectionDeferred = Deferred()
        clientProtocol = ClosingLaterProtocol(lostConnectionDeferred)
        clientRef = ref(clientProtocol)
        reactor = self.buildReactor()
        portDeferred = self.endpoints.server(reactor).listen(ServerFactory.forProtocol(Protocol))

        def listening(port):
            msg(f'Listening on {port.getHost()!r}')
            endpoint = self.endpoints.client(reactor, port.getHost())
            client = endpoint.connect(ClientFactory.forProtocol(lambda: clientProtocol))

            def disconnect(proto):
                msg(f'About to disconnect {proto!r}')
                proto.transport.loseConnection()
            client.addCallback(disconnect)
            client.addErrback(lostConnectionDeferred.errback)
            return lostConnectionDeferred

        def onListening():
            portDeferred.addCallback(listening)
            portDeferred.addErrback(err)
            portDeferred.addBoth(lambda ignored: reactor.stop())
        needsRunningReactor(reactor, onListening)
        self.runReactor(reactor)
        clientProtocol = None
        collect()
        self.assertIsNone(clientRef())
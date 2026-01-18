import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_sendPackets(self):
    """
        Datagrams can be sent with the transport's C{write} method and
        received via the C{datagramReceived} callback method.
        """
    server = Server()
    serverStarted = server.startedDeferred = defer.Deferred()
    port1 = reactor.listenUDP(0, server, interface='127.0.0.1')
    client = GoodClient()
    clientStarted = client.startedDeferred = defer.Deferred()

    def cbServerStarted(ignored):
        self.port2 = reactor.listenUDP(0, client, interface='127.0.0.1')
        return clientStarted
    d = serverStarted.addCallback(cbServerStarted)

    def cbClientStarted(ignored):
        client.transport.connect('127.0.0.1', server.transport.getHost().port)
        cAddr = client.transport.getHost()
        sAddr = server.transport.getHost()
        serverSend = client.packetReceived = defer.Deferred()
        server.transport.write(b'hello', (cAddr.host, cAddr.port))
        clientWrites = [(b'a',), (b'b', None), (b'c', (sAddr.host, sAddr.port))]

        def cbClientSend(ignored):
            if clientWrites:
                nextClientWrite = server.packetReceived = defer.Deferred()
                nextClientWrite.addCallback(cbClientSend)
                client.transport.write(*clientWrites.pop(0))
                return nextClientWrite
        return defer.DeferredList([cbClientSend(None), serverSend], fireOnOneErrback=True)
    d.addCallback(cbClientStarted)

    def cbSendsFinished(ignored):
        cAddr = client.transport.getHost()
        sAddr = server.transport.getHost()
        self.assertEqual(client.packets, [(b'hello', (sAddr.host, sAddr.port))])
        clientAddr = (cAddr.host, cAddr.port)
        self.assertEqual(server.packets, [(b'a', clientAddr), (b'b', clientAddr), (b'c', clientAddr)])
    d.addCallback(cbSendsFinished)

    def cbFinished(ignored):
        return defer.DeferredList([defer.maybeDeferred(port1.stopListening), defer.maybeDeferred(self.port2.stopListening)], fireOnOneErrback=True)
    d.addCallback(cbFinished)
    return d
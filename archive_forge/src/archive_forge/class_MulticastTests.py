import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
@skipIf(not interfaces.IReactorMulticast(reactor, None), 'This reactor does not support multicast')
class MulticastTests(TestCase):
    if os.environ.get('INFRASTRUCTURE') == 'AZUREPIPELINES' and runtime.platform.isMacOSX():
        skip = 'Does not work on Azure Pipelines'
    if not interfaces.IReactorMulticast(reactor, None):
        skip = 'This reactor does not support multicast'

    def setUp(self):
        self.server = Server()
        self.client = Client()
        self.port1 = reactor.listenMulticast(0, self.server)
        self.port2 = reactor.listenMulticast(0, self.client)
        self.client.transport.connect('127.0.0.1', self.server.transport.getHost().port)

    def tearDown(self):
        return gatherResults([maybeDeferred(self.port1.stopListening), maybeDeferred(self.port2.stopListening)])

    def testTTL(self):
        for o in (self.client, self.server):
            self.assertEqual(o.transport.getTTL(), 1)
            o.transport.setTTL(2)
            self.assertEqual(o.transport.getTTL(), 2)

    def test_loopback(self):
        """
        Test that after loopback mode has been set, multicast packets are
        delivered to their sender.
        """
        self.assertEqual(self.server.transport.getLoopbackMode(), 1)
        addr = self.server.transport.getHost()
        joined = self.server.transport.joinGroup('225.0.0.250')

        def cbJoined(ignored):
            d = self.server.packetReceived = Deferred()
            self.server.transport.write(b'hello', ('225.0.0.250', addr.port))
            return d
        joined.addCallback(cbJoined)

        def cbPacket(ignored):
            self.assertEqual(len(self.server.packets), 1)
            self.server.transport.setLoopbackMode(0)
            self.assertEqual(self.server.transport.getLoopbackMode(), 0)
            self.server.transport.write(b'hello', ('225.0.0.250', addr.port))
            d = Deferred()
            reactor.callLater(0, d.callback, None)
            return d
        joined.addCallback(cbPacket)

        def cbNoPacket(ignored):
            self.assertEqual(len(self.server.packets), 1)
        joined.addCallback(cbNoPacket)
        return joined

    def test_interface(self):
        """
        Test C{getOutgoingInterface} and C{setOutgoingInterface}.
        """
        self.assertEqual(self.client.transport.getOutgoingInterface(), '0.0.0.0')
        self.assertEqual(self.server.transport.getOutgoingInterface(), '0.0.0.0')
        d1 = self.client.transport.setOutgoingInterface('127.0.0.1')
        d2 = self.server.transport.setOutgoingInterface('127.0.0.1')
        result = gatherResults([d1, d2])

        def cbInterfaces(ignored):
            self.assertEqual(self.client.transport.getOutgoingInterface(), '127.0.0.1')
            self.assertEqual(self.server.transport.getOutgoingInterface(), '127.0.0.1')
        result.addCallback(cbInterfaces)
        return result

    def test_joinLeave(self):
        """
        Test that multicast a group can be joined and left.
        """
        d = self.client.transport.joinGroup('225.0.0.250')

        def clientJoined(ignored):
            return self.client.transport.leaveGroup('225.0.0.250')
        d.addCallback(clientJoined)

        def clientLeft(ignored):
            return self.server.transport.joinGroup('225.0.0.250')
        d.addCallback(clientLeft)

        def serverJoined(ignored):
            return self.server.transport.leaveGroup('225.0.0.250')
        d.addCallback(serverJoined)
        return d

    @skipIf(runtime.platform.isWindows() and (not runtime.platform.isVista()), "Windows' UDP multicast is not yet fully supported.")
    def test_joinFailure(self):
        """
        Test that an attempt to join an address which is not a multicast
        address fails with L{error.MulticastJoinError}.
        """
        return self.assertFailure(self.client.transport.joinGroup('127.0.0.1'), error.MulticastJoinError)

    def test_multicast(self):
        """
        Test that a multicast group can be joined and messages sent to and
        received from it.
        """
        c = Server()
        p = reactor.listenMulticast(0, c)
        addr = self.server.transport.getHost()
        joined = self.server.transport.joinGroup('225.0.0.250')

        def cbJoined(ignored):
            d = self.server.packetReceived = Deferred()
            c.transport.write(b'hello world', ('225.0.0.250', addr.port))
            return d
        joined.addCallback(cbJoined)

        def cbPacket(ignored):
            self.assertEqual(self.server.packets[0][0], b'hello world')
        joined.addCallback(cbPacket)

        def cleanup(passthrough):
            result = maybeDeferred(p.stopListening)
            result.addCallback(lambda ign: passthrough)
            return result
        joined.addCallback(cleanup)
        return joined

    @skipIf(runtime.platform.isWindows(), 'on non-linux platforms it appears multiple processes can listen, but not multiple sockets in same process?')
    def test_multiListen(self):
        """
        Test that multiple sockets can listen on the same multicast port and
        that they both receive multicast messages directed to that address.
        """
        firstClient = Server()
        firstPort = reactor.listenMulticast(0, firstClient, listenMultiple=True)
        portno = firstPort.getHost().port
        secondClient = Server()
        secondPort = reactor.listenMulticast(portno, secondClient, listenMultiple=True)
        theGroup = '225.0.0.250'
        joined = gatherResults([self.server.transport.joinGroup(theGroup), firstPort.joinGroup(theGroup), secondPort.joinGroup(theGroup)])

        def serverJoined(ignored):
            d1 = firstClient.packetReceived = Deferred()
            d2 = secondClient.packetReceived = Deferred()
            firstClient.transport.write(b'hello world', (theGroup, portno))
            return gatherResults([d1, d2])
        joined.addCallback(serverJoined)

        def gotPackets(ignored):
            self.assertEqual(firstClient.packets[0][0], b'hello world')
            self.assertEqual(secondClient.packets[0][0], b'hello world')
        joined.addCallback(gotPackets)

        def cleanup(passthrough):
            result = gatherResults([maybeDeferred(firstPort.stopListening), maybeDeferred(secondPort.stopListening)])
            result.addCallback(lambda ign: passthrough)
            return result
        joined.addBoth(cleanup)
        return joined
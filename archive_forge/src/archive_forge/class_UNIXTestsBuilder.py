from hashlib import md5
from os import close, fstat, stat, unlink, urandom
from pprint import pformat
from socket import AF_INET, SOCK_STREAM, SOL_SOCKET, socket
from stat import S_IMODE
from struct import pack
from tempfile import mkstemp, mktemp
from typing import Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from twisted.internet import base, interfaces
from twisted.internet.address import UNIXAddress
from twisted.internet.defer import Deferred, fail, gatherResults
from twisted.internet.endpoints import UNIXClientEndpoint, UNIXServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, DatagramProtocol, ServerFactory
from twisted.internet.task import LoopingCall
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.python.log import addObserver, err, removeObserver
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
class UNIXTestsBuilder(UNIXFamilyMixin, ReactorBuilder, ConnectionTestsMixin):
    """
    Builder defining tests relating to L{IReactorUNIX}.
    """
    requiredInterfaces = (IReactorUNIX,)
    endpoints = UNIXCreator()

    def test_mode(self):
        """
        The UNIX socket created by L{IReactorUNIX.listenUNIX} is created with
        the mode specified.
        """
        self._modeTest('listenUNIX', self.mktemp(), ServerFactory())

    @skipIf(not platform.isLinux(), 'Abstract namespace UNIX sockets only supported on Linux.')
    def test_listenOnLinuxAbstractNamespace(self):
        """
        On Linux, a UNIX socket path may begin with C{'\x00'} to indicate
        a socket in the abstract namespace.  L{IReactorUNIX.listenUNIX}
        accepts such a path.
        """
        path = _abstractPath(self)
        reactor = self.buildReactor()
        port = reactor.listenUNIX('\x00' + path, ServerFactory())
        self.assertEqual(port.getHost(), UNIXAddress('\x00' + path))

    def test_listenFailure(self):
        """
        L{IReactorUNIX.listenUNIX} raises L{CannotListenError} if the
        underlying port's createInternetSocket raises a socket error.
        """

        def raiseSocketError(self):
            raise OSError('FakeBasePort forced socket.error')
        self.patch(base.BasePort, 'createInternetSocket', raiseSocketError)
        reactor = self.buildReactor()
        with self.assertRaises(CannotListenError):
            reactor.listenUNIX('not-used', ServerFactory())

    @skipIf(not platform.isLinux(), 'Abstract namespace UNIX sockets only supported on Linux.')
    def test_connectToLinuxAbstractNamespace(self):
        """
        L{IReactorUNIX.connectUNIX} also accepts a Linux abstract namespace
        path.
        """
        path = _abstractPath(self)
        reactor = self.buildReactor()
        connector = reactor.connectUNIX('\x00' + path, ClientFactory())
        self.assertEqual(connector.getDestination(), UNIXAddress('\x00' + path))

    def test_addresses(self):
        """
        A client's transport's C{getHost} and C{getPeer} return L{UNIXAddress}
        instances which have the filesystem path of the host and peer ends of
        the connection.
        """

        class SaveAddress(ConnectableProtocol):

            def makeConnection(self, transport):
                self.addresses = dict(host=transport.getHost(), peer=transport.getPeer())
                transport.loseConnection()
        server = SaveAddress()
        client = SaveAddress()
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertEqual(server.addresses['host'], client.addresses['peer'])
        self.assertEqual(server.addresses['peer'], client.addresses['host'])

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_sendFileDescriptor(self):
        """
        L{IUNIXTransport.sendFileDescriptor} accepts an integer file descriptor
        and sends a copy of it to the process reading from the connection.
        """
        from socket import fromfd
        s = socket()
        s.bind(('', 0))
        server = SendFileDescriptor(s.fileno(), b'junk')
        client = ReceiveFileDescriptor()
        d = client.waitForDescriptor()

        def checkDescriptor(descriptor):
            received = fromfd(descriptor, AF_INET, SOCK_STREAM)
            close(descriptor)
            self.assertEqual(s.getsockname(), received.getsockname())
            self.assertNotEqual(s.fileno(), received.fileno())
        d.addCallback(checkDescriptor)
        d.addErrback(err, 'Sending file descriptor encountered a problem')
        d.addBoth(lambda ignored: server.transport.loseConnection())
        runProtocolsWithReactor(self, server, client, self.endpoints)

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_sendFileDescriptorTriggersPauseProducing(self):
        """
        If a L{IUNIXTransport.sendFileDescriptor} call fills up
        the send buffer, any registered producer is paused.
        """

        class DoesNotRead(ConnectableProtocol):

            def connectionMade(self):
                self.transport.pauseProducing()

        class SendsManyFileDescriptors(ConnectableProtocol):
            paused = False

            def connectionMade(self):
                self.socket = socket()
                self.transport.registerProducer(self, True)

                def sender():
                    self.transport.sendFileDescriptor(self.socket.fileno())
                    self.transport.write(b'x')
                self.task = LoopingCall(sender)
                self.task.clock = self.transport.reactor
                self.task.start(0).addErrback(err, 'Send loop failure')

            def stopProducing(self):
                self._disconnect()

            def resumeProducing(self):
                self._disconnect()

            def pauseProducing(self):
                self.paused = True
                self.transport.unregisterProducer()
                self._disconnect()

            def _disconnect(self):
                self.task.stop()
                self.transport.abortConnection()
                self.other.transport.abortConnection()
        server = SendsManyFileDescriptors()
        client = DoesNotRead()
        server.other = client
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertTrue(server.paused, 'sendFileDescriptor producer was not paused')

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_fileDescriptorOverrun(self):
        """
        If L{IUNIXTransport.sendFileDescriptor} is used to queue a greater
        number of file descriptors than the number of bytes sent using
        L{ITransport.write}, the connection is closed and the protocol connected
        to the transport has its C{connectionLost} method called with a failure
        wrapping L{FileDescriptorOverrun}.
        """
        cargo = socket()
        server = SendFileDescriptor(cargo.fileno(), None)
        client = ReceiveFileDescriptor()
        result = []
        d = client.waitForDescriptor()
        d.addBoth(result.append)
        d.addBoth(lambda ignored: server.transport.loseConnection())
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertIsInstance(result[0], Failure)
        result[0].trap(ConnectionClosed)
        self.assertIsInstance(server.reason.value, FileDescriptorOverrun)

    def _sendmsgMixinFileDescriptorReceivedDriver(self, ancillaryPacker):
        """
        Drive _SendmsgMixin via sendmsg socket calls to check that
        L{IFileDescriptorReceiver.fileDescriptorReceived} is called once
        for each file descriptor received in the ancillary messages.

        @param ancillaryPacker: A callable that will be given a list of
            two file descriptors and should return a two-tuple where:
            The first item is an iterable of zero or more (cmsg_level,
            cmsg_type, cmsg_data) tuples in the same order as the given
            list for actual sending via sendmsg; the second item is an
            integer indicating the expected number of FDs to be received.
        """
        from socket import socketpair
        from twisted.internet.unix import _SendmsgMixin
        from twisted.python.sendmsg import sendmsg

        def deviceInodeTuple(fd):
            fs = fstat(fd)
            return (fs.st_dev, fs.st_ino)

        @implementer(IFileDescriptorReceiver)
        class FakeProtocol(ConnectableProtocol):

            def __init__(self):
                self.fds = []
                self.deviceInodesReceived = []

            def fileDescriptorReceived(self, fd):
                self.fds.append(fd)
                self.deviceInodesReceived.append(deviceInodeTuple(fd))
                close(fd)

        class FakeReceiver(_SendmsgMixin):
            bufferSize = 1024

            def __init__(self, skt, proto):
                self.socket = skt
                self.protocol = proto

            def _dataReceived(self, data):
                pass

            def getHost(self):
                pass

            def getPeer(self):
                pass

            def _getLogPrefix(self, o):
                pass
        sendSocket, recvSocket = socketpair(AF_UNIX, SOCK_STREAM)
        self.addCleanup(sendSocket.close)
        self.addCleanup(recvSocket.close)
        proto = FakeProtocol()
        receiver = FakeReceiver(recvSocket, proto)
        fileOneFD, fileOneName = mkstemp()
        fileTwoFD, fileTwoName = mkstemp()
        self.addCleanup(unlink, fileOneName)
        self.addCleanup(unlink, fileTwoName)
        dataToSend = b'some data needs to be sent'
        fdsToSend = [fileOneFD, fileTwoFD]
        ancillary, expectedCount = ancillaryPacker(fdsToSend)
        sendmsg(sendSocket, dataToSend, ancillary)
        receiver.doRead()
        self.assertEqual(len(proto.fds), expectedCount)
        self.assertFalse(set(fdsToSend).intersection(set(proto.fds)))
        if proto.fds:
            deviceInodesSent = [deviceInodeTuple(fd) for fd in fdsToSend]
            self.assertEqual(deviceInodesSent, proto.deviceInodesReceived)

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_multiFileDescriptorReceivedPerRecvmsgOneCMSG(self):
        """
        _SendmsgMixin handles multiple file descriptors per recvmsg, calling
        L{IFileDescriptorReceiver.fileDescriptorReceived} once per received
        file descriptor. Scenario: single CMSG with two FDs.
        """
        from twisted.python.sendmsg import SCM_RIGHTS

        def ancillaryPacker(fdsToSend):
            ancillary = [(SOL_SOCKET, SCM_RIGHTS, pack('ii', *fdsToSend))]
            expectedCount = 2
            return (ancillary, expectedCount)
        self._sendmsgMixinFileDescriptorReceivedDriver(ancillaryPacker)

    @skipIf(platform.isMacOSX(), 'Multi control message ancillary sendmsg not supported on Mac.')
    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_multiFileDescriptorReceivedPerRecvmsgTwoCMSGs(self):
        """
        _SendmsgMixin handles multiple file descriptors per recvmsg, calling
        L{IFileDescriptorReceiver.fileDescriptorReceived} once per received
        file descriptor. Scenario: two CMSGs with one FD each.
        """
        from twisted.python.sendmsg import SCM_RIGHTS

        def ancillaryPacker(fdsToSend):
            ancillary = [(SOL_SOCKET, SCM_RIGHTS, pack('i', fd)) for fd in fdsToSend]
            expectedCount = 2
            return (ancillary, expectedCount)
        self._sendmsgMixinFileDescriptorReceivedDriver(ancillaryPacker)

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_multiFileDescriptorReceivedPerRecvmsgBadCMSG(self):
        """
        _SendmsgMixin handles multiple file descriptors per recvmsg, calling
        L{IFileDescriptorReceiver.fileDescriptorReceived} once per received
        file descriptor. Scenario: unsupported CMSGs.
        """
        from twisted.python import sendmsg

        def ancillaryPacker(fdsToSend):
            ancillary = []
            expectedCount = 0
            return (ancillary, expectedCount)

        def fakeRecvmsgUnsupportedAncillary(skt, *args, **kwargs):
            data = b'some data'
            ancillary = [(None, None, b'')]
            flags = 0
            return sendmsg.ReceivedMessage(data, ancillary, flags)
        events = []
        addObserver(events.append)
        self.addCleanup(removeObserver, events.append)
        self.patch(sendmsg, 'recvmsg', fakeRecvmsgUnsupportedAncillary)
        self._sendmsgMixinFileDescriptorReceivedDriver(ancillaryPacker)
        expectedMessage = 'received unsupported ancillary data'
        found = any((expectedMessage in e['format'] for e in events))
        self.assertTrue(found, 'Expected message not found in logged events')

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_avoidLeakingFileDescriptors(self):
        """
        If associated with a protocol which does not provide
        L{IFileDescriptorReceiver}, file descriptors received by the
        L{IUNIXTransport} implementation are closed and a warning is emitted.
        """
        from socket import socketpair
        probeClient, probeServer = socketpair()
        events = []
        addObserver(events.append)
        self.addCleanup(removeObserver, events.append)

        class RecordEndpointAddresses(SendFileDescriptor):

            def connectionMade(self):
                self.hostAddress = self.transport.getHost()
                self.peerAddress = self.transport.getPeer()
                SendFileDescriptor.connectionMade(self)
        server = RecordEndpointAddresses(probeClient.fileno(), b'junk')
        client = ConnectableProtocol()
        runProtocolsWithReactor(self, server, client, self.endpoints)
        probeClient.close()
        probeServer.setblocking(False)
        self.assertEqual(b'', probeServer.recv(1024))
        format = '%(protocolName)s (on %(hostAddress)r) does not provide IFileDescriptorReceiver; closing file descriptor received (from %(peerAddress)r).'
        clsName = 'ConnectableProtocol'
        expectedEvent = dict(hostAddress=server.peerAddress, peerAddress=server.hostAddress, protocolName=clsName, format=format)
        for logEvent in events:
            for k, v in expectedEvent.items():
                if v != logEvent.get(k):
                    break
            else:
                break
        else:
            self.fail('Expected event (%s) not found in logged events (%s)' % (expectedEvent, pformat(events)))

    @skipIf(not sendmsg, sendmsgSkipReason)
    def test_descriptorDeliveredBeforeBytes(self):
        """
        L{IUNIXTransport.sendFileDescriptor} sends file descriptors before
        L{ITransport.write} sends normal bytes.
        """

        @implementer(IFileDescriptorReceiver)
        class RecordEvents(ConnectableProtocol):

            def connectionMade(self):
                ConnectableProtocol.connectionMade(self)
                self.events = []

            def fileDescriptorReceived(innerSelf, descriptor):
                self.addCleanup(close, descriptor)
                innerSelf.events.append(type(descriptor))

            def dataReceived(self, data):
                self.events.extend(data)
        cargo = socket()
        server = SendFileDescriptor(cargo.fileno(), b'junk')
        client = RecordEvents()
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertEqual(int, client.events[0])
        self.assertEqual(b'junk', bytes(client.events[1:]))
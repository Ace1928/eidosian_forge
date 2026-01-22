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
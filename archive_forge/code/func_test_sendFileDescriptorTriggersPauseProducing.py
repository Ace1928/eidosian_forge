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
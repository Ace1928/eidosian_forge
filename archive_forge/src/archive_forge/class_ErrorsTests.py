from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
class ErrorsTests(unittest.SynchronousTestCase):
    """
    Error handling tests for C{udp.Port}.
    """

    def test_socketReadNormal(self) -> None:
        """
        Socket reads with some good data followed by a socket error which can
        be ignored causes reading to stop, and no log messages to be logged.
        """
        udp._sockErrReadIgnore.append(-7000)
        self.addCleanup(udp._sockErrReadIgnore.remove, -7000)
        protocol = KeepReads()
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'result', b'123', socket.error(-7000), b'456', socket.error(-7000)])
        port.doRead()
        self.assertEqual(protocol.reads, [b'result', b'123'])
        port.doRead()
        self.assertEqual(protocol.reads, [b'result', b'123', b'456'])

    def test_readImmediateError(self) -> None:
        """
        If the socket is unconnected, socket reads with an immediate
        connection refusal are ignored, and reading stops. The protocol's
        C{connectionRefused} method is not called.
        """
        udp._sockErrReadRefuse.append(-6000)
        self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
        protocol = KeepReads()
        protocol.connectionRefused = lambda: 1 / 0
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a'])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a', b'b'])

    def test_connectedReadImmediateError(self) -> None:
        """
        If the socket connected, socket reads with an immediate
        connection refusal are ignored, and reading stops. The protocol's
        C{connectionRefused} method is called.
        """
        udp._sockErrReadRefuse.append(-6000)
        self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
        protocol = KeepReads()
        refused = []
        protocol.connectionRefused = lambda: refused.append(True)
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
        port.connect('127.0.0.1', 9999)
        port.doRead()
        self.assertEqual(protocol.reads, [b'a'])
        self.assertEqual(refused, [True])
        port.doRead()
        self.assertEqual(protocol.reads, [b'a', b'b'])
        self.assertEqual(refused, [True])

    def test_readUnknownError(self) -> None:
        """
        Socket reads with an unknown socket error are raised.
        """
        protocol = KeepReads()
        port = udp.Port(None, protocol)
        port.socket = StringUDPSocket([b'good', socket.error(-1337)])
        self.assertRaises(socket.error, port.doRead)
        self.assertEqual(protocol.reads, [b'good'])
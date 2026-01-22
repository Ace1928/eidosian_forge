import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class BindTests(unittest.TestCase):
    """
    Tests for SOCKS and SOCKSv4a bind requests using the L{SOCKSv4} protocol.
    """

    def setUp(self):
        self.sock = SOCKSv4Driver()
        self.sock.transport = StringTCPTransport()
        self.sock.connectionMade()
        self.sock.reactor = FakeResolverReactor({b'localhost': '127.0.0.1'})

    def test_simple(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 1234) + socket.inet_aton('6.7.8.9'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNotNone(self.sock.driver_listen)
        incoming = self.sock.driver_listen.buildProtocol(('1.2.3.4', 5345))
        self.assertIsNotNone(incoming)
        incoming.transport = StringTCPTransport()
        incoming.connectionMade()
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 0) + socket.inet_aton('0.0.0.0'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(incoming.transport.value(), b'hello, world')
        incoming.dataReceived(b'hi there')
        self.assertEqual(self.sock.transport.value(), b'hi there')
        self.sock.connectionLost('fake reason')

    def test_socks4a(self):
        """
        If the destination IP address has zeros for the first three octets and
        non-zero for the fourth octet, the client is attempting a v4a
        connection.  A hostname is specified after the user ID string and the
        server connects to the address that hostname resolves to.

        @see: U{http://en.wikipedia.org/wiki/SOCKS#SOCKS_4a_protocol}
        """
        clientRequest = struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('0.0.0.1') + b'fooBAZ\x00' + b'localhost\x00'
        for byte in iterbytes(clientRequest):
            self.sock.dataReceived(byte)
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 1234) + socket.inet_aton('6.7.8.9'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNotNone(self.sock.driver_listen)
        incoming = self.sock.driver_listen.buildProtocol(('127.0.0.1', 5345))
        self.assertIsNotNone(incoming)
        incoming.transport = StringTCPTransport()
        incoming.connectionMade()
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 0) + socket.inet_aton('0.0.0.0'))
        self.assertIsNot(self.sock.transport.stringTCPTransport_closing, None)
        self.sock.dataReceived(b'hi there')
        self.assertEqual(incoming.transport.value(), b'hi there')
        incoming.dataReceived(b'hi there')
        self.assertEqual(self.sock.transport.value(), b'hi there')
        self.sock.connectionLost('fake reason')

    def test_socks4aFailedResolution(self):
        """
        Failed hostname resolution on a SOCKSv4a packet results in a 91 error
        response and the connection getting closed.
        """
        clientRequest = struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('0.0.0.1') + b'fooBAZ\x00' + b'failinghost\x00'
        for byte in iterbytes(clientRequest):
            self.sock.dataReceived(byte)
        sent = self.sock.transport.value()
        self.assertEqual(sent, struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
        self.assertTrue(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNone(self.sock.driver_outgoing)

    def test_accessDenied(self):
        self.sock.authorize = lambda code, server, port, user: 0
        self.sock.dataReceived(struct.pack('!BBH', 4, 2, 4242) + socket.inet_aton('10.2.3.4') + b'fooBAR' + b'\x00')
        self.assertEqual(self.sock.transport.value(), struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
        self.assertTrue(self.sock.transport.stringTCPTransport_closing)
        self.assertIsNone(self.sock.driver_listen)

    def test_eofRemote(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        incoming = self.sock.driver_listen.buildProtocol(('1.2.3.4', 5345))
        self.assertIsNotNone(incoming)
        incoming.transport = StringTCPTransport()
        incoming.connectionMade()
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 0) + socket.inet_aton('0.0.0.0'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(incoming.transport.value(), b'hello, world')
        incoming.transport.loseConnection()
        incoming.connectionLost('fake reason')

    def test_eofLocal(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        incoming = self.sock.driver_listen.buildProtocol(('1.2.3.4', 5345))
        self.assertIsNotNone(incoming)
        incoming.transport = StringTCPTransport()
        incoming.connectionMade()
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 90, 0) + socket.inet_aton('0.0.0.0'))
        self.assertFalse(self.sock.transport.stringTCPTransport_closing)
        self.sock.dataReceived(b'hello, world')
        self.assertEqual(incoming.transport.value(), b'hello, world')
        self.sock.connectionLost('fake reason')

    def test_badSource(self):
        self.sock.dataReceived(struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        incoming = self.sock.driver_listen.buildProtocol(('1.6.6.6', 666))
        self.assertIsNone(incoming)
        sent = self.sock.transport.value()
        self.sock.transport.clear()
        self.assertEqual(sent, struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
        self.assertTrue(self.sock.transport.stringTCPTransport_closing)
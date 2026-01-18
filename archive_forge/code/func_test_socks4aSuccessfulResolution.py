import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_socks4aSuccessfulResolution(self):
    """
        If the destination IP address has zeros for the first three octets and
        non-zero for the fourth octet, the client is attempting a v4a
        connection.  A hostname is specified after the user ID string and the
        server connects to the address that hostname resolves to.

        @see: U{http://en.wikipedia.org/wiki/SOCKS#SOCKS_4a_protocol}
        """
    clientRequest = struct.pack('!BBH', 4, 1, 34) + socket.inet_aton('0.0.0.1') + b'fooBAZ\x00' + b'localhost\x00'
    for byte in iterbytes(clientRequest):
        self.sock.dataReceived(byte)
    sent = self.sock.transport.value()
    self.sock.transport.clear()
    self.assertEqual(sent, struct.pack('!BBH', 0, 90, 34) + socket.inet_aton('127.0.0.1'))
    self.assertFalse(self.sock.transport.stringTCPTransport_closing)
    self.assertIsNotNone(self.sock.driver_outgoing)
    self.sock.dataReceived(b'hello, world')
    self.assertEqual(self.sock.driver_outgoing.transport.value(), b'hello, world')
    self.sock.driver_outgoing.dataReceived(b'hi there')
    self.assertEqual(self.sock.transport.value(), b'hi there')
    self.sock.connectionLost('fake reason')
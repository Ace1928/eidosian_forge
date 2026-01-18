import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
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
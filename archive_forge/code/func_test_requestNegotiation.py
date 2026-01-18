from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_requestNegotiation(self):
    """
        L{telnet.Telnet.requestNegotiation} formats the feature byte and the
        payload bytes into the subnegotiation format and sends them.

        See RFC 855.
        """
    transport = proto_helpers.StringTransport()
    self.protocol.makeConnection(transport)
    self.protocol.requestNegotiation(b'\x01', b'\x02\x03')
    self.assertEqual(transport.value(), b'\xff\xfa\x01\x02\x03\xff\xf0')
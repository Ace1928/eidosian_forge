import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class IntNTestCaseMixin(LPTestCaseMixin):
    """
    TestCase mixin for int-prefixed protocols.
    """
    protocol: 'Optional[Type[protocol.Protocol]]' = None
    strings: Optional[List[bytes]] = None
    illegalStrings: Optional[List[bytes]] = None
    partialStrings: Optional[List[bytes]] = None

    def test_receive(self):
        """
        Test receiving data find the same data send.
        """
        r = self.getProtocol()
        for s in self.strings:
            for c in iterbytes(struct.pack(r.structFormat, len(s)) + s):
                r.dataReceived(c)
        self.assertEqual(r.received, self.strings)

    def test_partial(self):
        """
        Send partial data, nothing should be definitely received.
        """
        for s in self.partialStrings:
            r = self.getProtocol()
            for c in iterbytes(s):
                r.dataReceived(c)
            self.assertEqual(r.received, [])

    def test_send(self):
        """
        Test sending data over protocol.
        """
        r = self.getProtocol()
        r.sendString(b'b' * 16)
        self.assertEqual(r.transport.value(), struct.pack(r.structFormat, 16) + b'b' * 16)

    def test_lengthLimitExceeded(self):
        """
        When a length prefix is received which is greater than the protocol's
        C{MAX_LENGTH} attribute, the C{lengthLimitExceeded} method is called
        with the received length prefix.
        """
        length = []
        r = self.getProtocol()
        r.lengthLimitExceeded = length.append
        r.MAX_LENGTH = 10
        r.dataReceived(struct.pack(r.structFormat, 11))
        self.assertEqual(length, [11])

    def test_longStringNotDelivered(self):
        """
        If a length prefix for a string longer than C{MAX_LENGTH} is delivered
        to C{dataReceived} at the same time as the entire string, the string is
        not passed to C{stringReceived}.
        """
        r = self.getProtocol()
        r.MAX_LENGTH = 10
        r.dataReceived(struct.pack(r.structFormat, 11) + b'x' * 11)
        self.assertEqual(r.received, [])

    def test_stringReceivedNotImplemented(self):
        """
        When L{IntNStringReceiver.stringReceived} is not overridden in a
        subclass, calling it raises C{NotImplementedError}.
        """
        proto = basic.IntNStringReceiver()
        self.assertRaises(NotImplementedError, proto.stringReceived, 'foo')
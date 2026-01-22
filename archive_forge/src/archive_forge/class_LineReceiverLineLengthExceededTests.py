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
class LineReceiverLineLengthExceededTests(unittest.SynchronousTestCase):
    """
    Tests for L{twisted.protocols.basic.LineReceiver.lineLengthExceeded}.
    """

    def setUp(self):
        self.proto = ExcessivelyLargeLineCatcher()
        self.proto.MAX_LENGTH = 6
        self.transport = proto_helpers.StringTransport()
        self.proto.makeConnection(self.transport)

    def test_longUnendedLine(self):
        """
        If more bytes than C{LineReceiver.MAX_LENGTH} arrive containing no line
        delimiter, all of the bytes are passed as a single string to
        L{LineReceiver.lineLengthExceeded}.
        """
        excessive = b'x' * (self.proto.MAX_LENGTH * 2 + 2)
        self.proto.dataReceived(excessive)
        self.assertEqual([excessive], self.proto.longLines)

    def test_longLineAfterShortLine(self):
        """
        If L{LineReceiver.dataReceived} is called with bytes representing a
        short line followed by bytes that exceed the length limit without a
        line delimiter, L{LineReceiver.lineLengthExceeded} is called with all
        of the bytes following the short line's delimiter.
        """
        excessive = b'x' * (self.proto.MAX_LENGTH * 2 + 2)
        self.proto.dataReceived(b'x' + self.proto.delimiter + excessive)
        self.assertEqual([excessive], self.proto.longLines)

    def test_longLineWithDelimiter(self):
        """
        If L{LineReceiver.dataReceived} is called with more than
        C{LineReceiver.MAX_LENGTH} bytes containing a line delimiter somewhere
        not in the first C{MAX_LENGTH} bytes, the entire byte string is passed
        to L{LineReceiver.lineLengthExceeded}.
        """
        excessive = self.proto.delimiter.join([b'x' * (self.proto.MAX_LENGTH * 2 + 2)] * 2)
        self.proto.dataReceived(excessive)
        self.assertEqual([excessive], self.proto.longLines)

    def test_multipleLongLines(self):
        """
        If L{LineReceiver.dataReceived} is called with more than
        C{LineReceiver.MAX_LENGTH} bytes containing multiple line delimiters
        somewhere not in the first C{MAX_LENGTH} bytes, the entire byte string
        is passed to L{LineReceiver.lineLengthExceeded}.
        """
        excessive = (b'x' * (self.proto.MAX_LENGTH * 2 + 2) + self.proto.delimiter) * 2
        self.proto.dataReceived(excessive)
        self.assertEqual([excessive], self.proto.longLines)

    def test_maximumLineLength(self):
        """
        C{LineReceiver} disconnects the transport if it receives a line longer
        than its C{MAX_LENGTH}.
        """
        proto = basic.LineReceiver()
        transport = proto_helpers.StringTransport()
        proto.makeConnection(transport)
        proto.dataReceived(b'x' * (proto.MAX_LENGTH + 1) + b'\r\nr')
        self.assertTrue(transport.disconnecting)

    def test_maximumLineLengthRemaining(self):
        """
        C{LineReceiver} disconnects the transport it if receives a non-finished
        line longer than its C{MAX_LENGTH}.
        """
        proto = basic.LineReceiver()
        transport = proto_helpers.StringTransport()
        proto.makeConnection(transport)
        proto.dataReceived(b'x' * (proto.MAX_LENGTH + len(proto.delimiter)))
        self.assertTrue(transport.disconnecting)
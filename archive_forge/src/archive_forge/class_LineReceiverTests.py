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
class LineReceiverTests(unittest.SynchronousTestCase):
    """
    Test L{twisted.protocols.basic.LineReceiver}, using the C{LineTester}
    wrapper.
    """
    buffer = b'len 10\n\n0123456789len 5\n\n1234\nlen 20\nfoo 123\n\n0123456789\n012345678len 0\nfoo 5\n\n1234567890123456789012345678901234567890123456789012345678901234567890\nlen 1\n\na'
    output = [b'len 10', b'0123456789', b'len 5', b'1234\n', b'len 20', b'foo 123', b'0123456789\n012345678', b'len 0', b'foo 5', b'', b'67890', b'len 1', b'a']

    def test_buffer(self):
        """
        Test buffering for different packet size, checking received matches
        expected data.
        """
        for packet_size in range(1, 10):
            t = proto_helpers.StringIOWithoutClosing()
            a = LineTester()
            a.makeConnection(protocol.FileWrapper(t))
            for i in range(len(self.buffer) // packet_size + 1):
                s = self.buffer[i * packet_size:(i + 1) * packet_size]
                a.dataReceived(s)
            self.assertEqual(self.output, a.received)
    pauseBuf = b'twiddle1\ntwiddle2\npause\ntwiddle3\n'
    pauseOutput1 = [b'twiddle1', b'twiddle2', b'pause']
    pauseOutput2 = pauseOutput1 + [b'twiddle3']

    def test_pausing(self):
        """
        Test pause inside data receiving. It uses fake clock to see if
        pausing/resuming work.
        """
        for packet_size in range(1, 10):
            t = proto_helpers.StringIOWithoutClosing()
            clock = task.Clock()
            a = LineTester(clock)
            a.makeConnection(protocol.FileWrapper(t))
            for i in range(len(self.pauseBuf) // packet_size + 1):
                s = self.pauseBuf[i * packet_size:(i + 1) * packet_size]
                a.dataReceived(s)
            self.assertEqual(self.pauseOutput1, a.received)
            clock.advance(0)
            self.assertEqual(self.pauseOutput2, a.received)
    rawpauseBuf = b'twiddle1\ntwiddle2\nlen 5\nrawpause\n12345twiddle3\n'
    rawpauseOutput1 = [b'twiddle1', b'twiddle2', b'len 5', b'rawpause', b'']
    rawpauseOutput2 = [b'twiddle1', b'twiddle2', b'len 5', b'rawpause', b'12345', b'twiddle3']

    def test_rawPausing(self):
        """
        Test pause inside raw date receiving.
        """
        for packet_size in range(1, 10):
            t = proto_helpers.StringIOWithoutClosing()
            clock = task.Clock()
            a = LineTester(clock)
            a.makeConnection(protocol.FileWrapper(t))
            for i in range(len(self.rawpauseBuf) // packet_size + 1):
                s = self.rawpauseBuf[i * packet_size:(i + 1) * packet_size]
                a.dataReceived(s)
            self.assertEqual(self.rawpauseOutput1, a.received)
            clock.advance(0)
            self.assertEqual(self.rawpauseOutput2, a.received)
    stop_buf = b'twiddle1\ntwiddle2\nstop\nmore\nstuff\n'
    stop_output = [b'twiddle1', b'twiddle2', b'stop']

    def test_stopProducing(self):
        """
        Test stop inside producing.
        """
        for packet_size in range(1, 10):
            t = proto_helpers.StringIOWithoutClosing()
            a = LineTester()
            a.makeConnection(protocol.FileWrapper(t))
            for i in range(len(self.stop_buf) // packet_size + 1):
                s = self.stop_buf[i * packet_size:(i + 1) * packet_size]
                a.dataReceived(s)
            self.assertEqual(self.stop_output, a.received)

    def test_lineReceiverAsProducer(self):
        """
        Test produce/unproduce in receiving.
        """
        a = LineTester()
        t = proto_helpers.StringIOWithoutClosing()
        a.makeConnection(protocol.FileWrapper(t))
        a.dataReceived(b'produce\nhello world\nunproduce\ngoodbye\n')
        self.assertEqual(a.received, [b'produce', b'hello world', b'unproduce', b'goodbye'])

    def test_clearLineBuffer(self):
        """
        L{LineReceiver.clearLineBuffer} removes all buffered data and returns
        it as a C{bytes} and can be called from beneath C{dataReceived}.
        """

        class ClearingReceiver(basic.LineReceiver):

            def lineReceived(self, line):
                self.line = line
                self.rest = self.clearLineBuffer()
        protocol = ClearingReceiver()
        protocol.dataReceived(b'foo\r\nbar\r\nbaz')
        self.assertEqual(protocol.line, b'foo')
        self.assertEqual(protocol.rest, b'bar\r\nbaz')
        protocol.dataReceived(b'quux\r\n')
        self.assertEqual(protocol.line, b'quux')
        self.assertEqual(protocol.rest, b'')

    def test_stackRecursion(self):
        """
        Test switching modes many times on the same data.
        """
        proto = FlippingLineTester()
        transport = proto_helpers.StringIOWithoutClosing()
        proto.makeConnection(protocol.FileWrapper(transport))
        limit = sys.getrecursionlimit()
        proto.dataReceived(b'x\nx' * limit)
        self.assertEqual(b'x' * limit, b''.join(proto.lines))

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

    def test_maximumLineLengthPartialDelimiter(self):
        """
        C{LineReceiver} doesn't disconnect the transport when it
        receives a finished line as long as its C{MAX_LENGTH}, when
        the second-to-last packet ended with a pattern that could have
        been -- and turns out to have been -- the start of a
        delimiter, and that packet causes the total input to exceed
        C{MAX_LENGTH} + len(delimiter).
        """
        proto = LineTester()
        proto.MAX_LENGTH = 4
        t = proto_helpers.StringTransport()
        proto.makeConnection(t)
        line = b'x' * (proto.MAX_LENGTH - 1)
        proto.dataReceived(line)
        proto.dataReceived(proto.delimiter[:-1])
        proto.dataReceived(proto.delimiter[-1:] + line)
        self.assertFalse(t.disconnecting)
        self.assertEqual(len(proto.received), 1)
        self.assertEqual(line, proto.received[0])

    def test_notQuiteMaximumLineLengthUnfinished(self):
        """
        C{LineReceiver} doesn't disconnect the transport it if
        receives a non-finished line whose length, counting the
        delimiter, is longer than its C{MAX_LENGTH} but shorter than
        its C{MAX_LENGTH} + len(delimiter). (When the first part that
        exceeds the max is the beginning of the delimiter.)
        """
        proto = basic.LineReceiver()
        proto.delimiter = b'\r\n'
        transport = proto_helpers.StringTransport()
        proto.makeConnection(transport)
        proto.dataReceived(b'x' * proto.MAX_LENGTH + proto.delimiter[:len(proto.delimiter) - 1])
        self.assertFalse(transport.disconnecting)

    def test_rawDataError(self):
        """
        C{LineReceiver.dataReceived} forwards errors returned by
        C{rawDataReceived}.
        """
        proto = basic.LineReceiver()
        proto.rawDataReceived = lambda data: RuntimeError('oops')
        transport = proto_helpers.StringTransport()
        proto.makeConnection(transport)
        proto.setRawMode()
        why = proto.dataReceived(b'data')
        self.assertIsInstance(why, RuntimeError)

    def test_rawDataReceivedNotImplemented(self):
        """
        When L{LineReceiver.rawDataReceived} is not overridden in a
        subclass, calling it raises C{NotImplementedError}.
        """
        proto = basic.LineReceiver()
        self.assertRaises(NotImplementedError, proto.rawDataReceived, 'foo')

    def test_lineReceivedNotImplemented(self):
        """
        When L{LineReceiver.lineReceived} is not overridden in a subclass,
        calling it raises C{NotImplementedError}.
        """
        proto = basic.LineReceiver()
        self.assertRaises(NotImplementedError, proto.lineReceived, 'foo')
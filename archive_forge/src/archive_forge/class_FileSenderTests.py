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
class FileSenderTests(unittest.TestCase):
    """
    Tests for L{basic.FileSender}.
    """

    def test_interface(self):
        """
        L{basic.FileSender} implements the L{IPullProducer} interface.
        """
        sender = basic.FileSender()
        self.assertTrue(verifyObject(IProducer, sender))

    def test_producerRegistered(self):
        """
        When L{basic.FileSender.beginFileTransfer} is called, it registers
        itself with provided consumer, as a non-streaming producer.
        """
        source = BytesIO(b'Test content')
        consumer = proto_helpers.StringTransport()
        sender = basic.FileSender()
        sender.beginFileTransfer(source, consumer)
        self.assertEqual(consumer.producer, sender)
        self.assertFalse(consumer.streaming)

    def test_transfer(self):
        """
        L{basic.FileSender} sends the content of the given file using a
        C{IConsumer} interface via C{beginFileTransfer}. It returns a
        L{Deferred} which fires with the last byte sent.
        """
        source = BytesIO(b'Test content')
        consumer = proto_helpers.StringTransport()
        sender = basic.FileSender()
        d = sender.beginFileTransfer(source, consumer)
        sender.resumeProducing()
        sender.resumeProducing()
        self.assertIsNone(consumer.producer)
        self.assertEqual(b't', self.successResultOf(d))
        self.assertEqual(b'Test content', consumer.value())

    def test_transferMultipleChunks(self):
        """
        L{basic.FileSender} reads at most C{CHUNK_SIZE} every time it resumes
        producing.
        """
        source = BytesIO(b'Test content')
        consumer = proto_helpers.StringTransport()
        sender = basic.FileSender()
        sender.CHUNK_SIZE = 4
        d = sender.beginFileTransfer(source, consumer)
        sender.resumeProducing()
        self.assertEqual(b'Test', consumer.value())
        sender.resumeProducing()
        self.assertEqual(b'Test con', consumer.value())
        sender.resumeProducing()
        self.assertEqual(b'Test content', consumer.value())
        sender.resumeProducing()
        self.assertEqual(b't', self.successResultOf(d))
        self.assertEqual(b'Test content', consumer.value())

    def test_transferWithTransform(self):
        """
        L{basic.FileSender.beginFileTransfer} takes a C{transform} argument
        which allows to manipulate the data on the fly.
        """

        def transform(chunk):
            return chunk.swapcase()
        source = BytesIO(b'Test content')
        consumer = proto_helpers.StringTransport()
        sender = basic.FileSender()
        d = sender.beginFileTransfer(source, consumer, transform)
        sender.resumeProducing()
        sender.resumeProducing()
        self.assertEqual(b'T', self.successResultOf(d))
        self.assertEqual(b'tEST CONTENT', consumer.value())

    def test_abortedTransfer(self):
        """
        The C{Deferred} returned by L{basic.FileSender.beginFileTransfer} fails
        with an C{Exception} if C{stopProducing} when the transfer is not
        complete.
        """
        source = BytesIO(b'Test content')
        consumer = proto_helpers.StringTransport()
        sender = basic.FileSender()
        d = sender.beginFileTransfer(source, consumer)
        sender.stopProducing()
        failure = self.failureResultOf(d)
        failure.trap(Exception)
        self.assertEqual('Consumer asked us to stop producing', str(failure.value))
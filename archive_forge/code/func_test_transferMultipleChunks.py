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
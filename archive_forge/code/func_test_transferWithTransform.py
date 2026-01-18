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
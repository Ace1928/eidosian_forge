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
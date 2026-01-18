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
def test_pauseResume(self):
    """
        When L{basic.LineReceiver} is paused, it doesn't deliver lines to
        L{basic.LineReceiver.lineReceived} and delivers them immediately upon
        being resumed.

        L{ConsumingProtocol} is a L{LineReceiver} that pauses itself after
        every line, and writes that line to its transport.
        """
    p = ConsumingProtocol()
    t = OnlyProducerTransport()
    p.makeConnection(t)
    p.dataReceived(b'hello, ')
    self.assertEqual(t.data, [])
    self.assertFalse(t.paused)
    self.assertFalse(p.paused)
    p.dataReceived(b'world\r\n')
    self.assertEqual(t.data, [b'hello, world'])
    self.assertTrue(t.paused)
    self.assertTrue(p.paused)
    p.resumeProducing()
    self.assertEqual(t.data, [b'hello, world'])
    self.assertFalse(t.paused)
    self.assertFalse(p.paused)
    p.dataReceived(b'hello\r\nworld\r\n')
    self.assertEqual(t.data, [b'hello, world', b'hello'])
    self.assertTrue(t.paused)
    self.assertTrue(p.paused)
    p.resumeProducing()
    self.assertEqual(t.data, [b'hello, world', b'hello', b'world'])
    self.assertTrue(t.paused)
    self.assertTrue(p.paused)
    p.dataReceived(b'goodbye\r\n')
    self.assertEqual(t.data, [b'hello, world', b'hello', b'world'])
    self.assertTrue(t.paused)
    self.assertTrue(p.paused)
    p.resumeProducing()
    self.assertEqual(t.data, [b'hello, world', b'hello', b'world', b'goodbye'])
    self.assertTrue(t.paused)
    self.assertTrue(p.paused)
    p.resumeProducing()
    self.assertEqual(t.data, [b'hello, world', b'hello', b'world', b'goodbye'])
    self.assertFalse(t.paused)
    self.assertFalse(p.paused)
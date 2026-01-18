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
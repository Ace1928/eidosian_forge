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
def test_receiveNetstringPortions_1(self):
    """
        Netstrings can be received in two portions.
        """
    self.netstringReceiver.dataReceived(b'4:aa')
    self.netstringReceiver.dataReceived(b'aa,')
    self.assertEqual(self.netstringReceiver.received, [b'aaaa'])
    self.assertTrue(self.netstringReceiver._payloadComplete())
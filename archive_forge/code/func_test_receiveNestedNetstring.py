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
def test_receiveNestedNetstring(self):
    """
        Netstrings with embedded netstrings. This test makes sure that
        the parser does not become confused about the ',' and ':'
        characters appearing inside the data portion of the netstring.
        """
    self.netstringReceiver.dataReceived(b'4:1:a,,')
    self.assertEqual(self.netstringReceiver.received, [b'1:a,'])
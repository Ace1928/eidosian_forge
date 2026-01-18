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
def test_recvdContainsRemainingData(self):
    """
        In stringReceived, recvd contains the remaining data that was passed to
        dataReceived that was not part of the current message.
        """
    result = []
    r = self.getProtocol()

    def stringReceived(receivedString):
        result.append(r.recvd)
    r.stringReceived = stringReceived
    completeMessage = struct.pack(r.structFormat, 5) + b'a' * 5
    incompleteMessage = struct.pack(r.structFormat, 5) + b'b' * 4
    r.dataReceived(completeMessage + incompleteMessage)
    self.assertEqual(result, [incompleteMessage])
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
def test_recvdInLengthLimitExceeded(self):
    """
        The L{IntNStringReceiver.recvd} buffer contains all data not yet
        processed by L{IntNStringReceiver.dataReceived} if the
        C{lengthLimitExceeded} event occurs.
        """
    proto = self.getProtocol()
    DATA = b'too long'
    proto.MAX_LENGTH = len(DATA) - 1
    message = self.makeMessage(proto, DATA)
    result = []

    def lengthLimitExceeded(length):
        result.append(length)
        result.append(proto.recvd)
    proto.lengthLimitExceeded = lengthLimitExceeded
    proto.dataReceived(message)
    self.assertEqual(result[0], len(DATA))
    self.assertEqual(result[1], message)
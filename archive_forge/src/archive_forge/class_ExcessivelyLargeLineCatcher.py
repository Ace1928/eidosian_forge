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
class ExcessivelyLargeLineCatcher(basic.LineReceiver):
    """
    Helper for L{LineReceiverLineLengthExceededTests}.

    @ivar longLines: A L{list} of L{bytes} giving the values
        C{lineLengthExceeded} has been called with.
    """

    def connectionMade(self):
        self.longLines = []

    def lineReceived(self, line):
        """
        Disregard any received lines.
        """

    def lineLengthExceeded(self, data):
        """
        Record any data that exceeds the line length limits.
        """
        self.longLines.append(data)
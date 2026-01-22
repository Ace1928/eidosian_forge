from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
class _ParserHelper:
    """
    A box receiver which records all boxes received.
    """

    def __init__(self):
        self.boxes = []

    def getPeer(self):
        return 'string'

    def getHost(self):
        return 'string'
    disconnecting = False

    def startReceivingBoxes(self, sender):
        """
        No initialization is required.
        """

    def ampBoxReceived(self, box):
        self.boxes.append(box)

    @classmethod
    def parse(cls, fileObj):
        """
        Parse some amp data stored in a file.

        @param fileObj: a file-like object.

        @return: a list of AmpBoxes encoded in the given file.
        """
        parserHelper = cls()
        bbp = BinaryBoxProtocol(boxReceiver=parserHelper)
        bbp.makeConnection(parserHelper)
        bbp.dataReceived(fileObj.read())
        return parserHelper.boxes

    @classmethod
    def parseString(cls, data):
        """
        Parse some amp data stored in a string.

        @param data: a str holding some amp-encoded data.

        @return: a list of AmpBoxes encoded in the given string.
        """
        return cls.parse(BytesIO(data))
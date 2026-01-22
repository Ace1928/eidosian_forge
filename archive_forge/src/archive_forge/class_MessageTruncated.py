import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class MessageTruncated:
    """
    An empty response message whose TR bit is set to 1.
    """

    @classmethod
    def bytes(cls):
        """
        Bytes which are expected when encoding an instance constructed using
        C{kwargs} and which are expected to result in an identical instance when
        decoded.

        @return: The L{bytes} of a wire encoded message.
        """
        return b'\x01\x00\x82\x00\x00\x00\x00\x00\x00\x00\x00\x00'

    @classmethod
    def kwargs(cls):
        """
        Keyword constructor arguments which are expected to result in an
        instance which returns C{bytes} when encoded.

        @return: A L{dict} of keyword arguments.
        """
        return dict(id=256, answer=1, opCode=0, auth=0, trunc=1, recDes=0, recAv=0, rCode=0, ednsVersion=None)
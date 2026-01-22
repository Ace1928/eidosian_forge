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
class MessageEDNSExtendedRCODE:
    """
    An example of an EDNS message with an extended RCODE.
    """

    @classmethod
    def bytes(cls):
        """
        Bytes which are expected when encoding an instance constructed using
        C{kwargs} and which are expected to result in an identical instance when
        decoded.

        @return: The L{bytes} of a wire encoded message.
        """
        return b'\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00)\x10\x00\xab\x00\x00\x00\x00\x00'

    @classmethod
    def kwargs(cls):
        """
        Keyword constructor arguments which are expected to result in an
        instance which returns C{bytes} when encoded.

        @return: A L{dict} of keyword arguments.
        """
        return dict(id=0, answer=False, opCode=dns.OP_QUERY, auth=False, trunc=False, recDes=False, recAv=False, rCode=2748, ednsVersion=0, dnssecOK=False, maxSize=4096, queries=[], answers=[], authority=[], additional=[])
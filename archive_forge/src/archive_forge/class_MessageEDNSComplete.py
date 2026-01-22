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
class MessageEDNSComplete:
    """
    An example of a fully populated edns response message.

    Contains name compression, answers, authority, and additional records.
    """

    @classmethod
    def bytes(cls):
        """
        Bytes which are expected when encoding an instance constructed using
        C{kwargs} and which are expected to result in an identical instance when
        decoded.

        @return: The L{bytes} of a wire encoded message.
        """
        return b"\x01\x00\x95\xbf\x00\x01\x00\x01\x00\x01\x00\x02\x07example\x03com\x00\x00\x06\x00\x01\xc0\x0c\x00\x06\x00\x01\xff\xff\xff\xff\x00'\x03ns1\xc0\x0c\nhostmaster\xc0\x0c\xff\xff\xff\xfe\x7f\xff\xff\xfd\x7f\xff\xff\xfc\x7f\xff\xff\xfb\xff\xff\xff\xfa\xc0\x0c\x00\x02\x00\x01\xff\xff\xff\xff\x00\x02\xc0)\xc0)\x00\x01\x00\x01\xff\xff\xff\xff\x00\x04\x05\x06\x07\x08\x00\x00)\x04\x00\x00\x03\x80\x00\x00\x00"

    @classmethod
    def kwargs(cls):
        """
        Keyword constructor arguments which are expected to result in an
        instance which returns C{bytes} when encoded.

        @return: A L{dict} of keyword arguments.
        """
        return dict(id=256, answer=1, opCode=dns.OP_STATUS, auth=1, trunc=0, recDes=1, recAv=1, rCode=15, ednsVersion=3, dnssecOK=True, authenticData=True, checkingDisabled=True, maxSize=1024, queries=[dns.Query(b'example.com', dns.SOA)], answers=[dns.RRHeader(b'example.com', type=dns.SOA, ttl=4294967295, auth=True, payload=dns.Record_SOA(ttl=4294967295, mname=b'ns1.example.com', rname=b'hostmaster.example.com', serial=4294967294, refresh=2147483645, retry=2147483644, expire=2147483643, minimum=4294967290))], authority=[dns.RRHeader(b'example.com', type=dns.NS, ttl=4294967295, auth=True, payload=dns.Record_NS('ns1.example.com', ttl=4294967295))], additional=[dns.RRHeader(b'ns1.example.com', type=dns.A, ttl=4294967295, auth=True, payload=dns.Record_A('5.6.7.8', ttl=4294967295))])
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
def test_optHeaderPosition(self):
    """
        L{dns._EDNSMessage} can decode OPT records, regardless of their position
        in the additional records section.

        "The OPT RR MAY be placed anywhere within the additional data section."

        @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.1}
        """
    b = BytesIO()
    optRecord = dns._OPTHeader(version=1)
    optRecord.encode(b)
    optRRHeader = dns.RRHeader()
    b.seek(0)
    optRRHeader.decode(b)
    m = dns.Message()
    m.additional = [optRRHeader]
    actualMessages = []
    actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
    m.additional.append(dns.RRHeader(type=dns.A))
    actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
    m.additional.insert(0, dns.RRHeader(type=dns.A))
    actualMessages.append(dns._EDNSMessage._fromMessage(m).ednsVersion)
    self.assertEqual([1] * 3, actualMessages)
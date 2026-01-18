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
def test_wks(self):
    """
        Two L{dns.Record_WKS} instances compare equal if and only if they have
        the same address, protocol, map, and ttl.
        """
    self._equalityTest(dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('4.3.2.1', 1, 'foo', 2))
    self._equalityTest(dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 100, 'foo', 2))
    self._equalityTest(dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'bar', 2))
    self._equalityTest(dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'foo', 2), dns.Record_WKS('1.2.3.4', 1, 'foo', 200))
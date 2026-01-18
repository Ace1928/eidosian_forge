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
def test_sshfp(self):
    """
        Two L{dns.Record_SSHFP} instances compare equal if and only if
        they have the same key type, fingerprint type, fingerprint, and ttl.
        """
    self._equalityTest(dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(2, 2, b'happyday', 40))
    self._equalityTest(dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 1, b'happyday', 40))
    self._equalityTest(dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happxday', 40))
    self._equalityTest(dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happyday', 40), dns.Record_SSHFP(1, 2, b'happyday', 45))
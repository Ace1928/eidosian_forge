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
def test_resourceRecordHeaderTypeMismatch(self):
    """
        L{RRHeader()} raises L{ValueError} when the given type and the type
        of the payload don't match.
        """
    with self.assertRaisesRegex(ValueError, 'Payload type \\(AAAA\\) .* type \\(A\\)'):
        dns.RRHeader(type=dns.A, payload=dns.Record_AAAA())
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
def test_responseType(self):
    """
        L{dns._responseFromMessage} returns a new instance of C{cls}
        """

    class SuppliedClass:
        id = 1
        queries = []
    expectedClass = dns.Message
    self.assertIsInstance(dns._responseFromMessage(responseConstructor=expectedClass, message=SuppliedClass()), expectedClass)
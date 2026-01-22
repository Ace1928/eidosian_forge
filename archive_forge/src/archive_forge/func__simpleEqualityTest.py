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
def _simpleEqualityTest(self, cls):
    """
        Assert that instances of C{cls} with the same attributes compare equal
        to each other and instances with different attributes compare as not
        equal.

        @param cls: A L{dns.SimpleRecord} subclass.
        """
    self._equalityTest(cls(b'example.com', 123), cls(b'example.com', 123), cls(b'example.com', 321))
    self._equalityTest(cls(b'example.com', 123), cls(b'example.com', 123), cls(b'example.org', 123))
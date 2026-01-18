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
def test_seconds(self):
    """
        Passed a string giving a number of seconds, L{dns.str2time} returns the
        number of seconds represented.  For example, C{"10S"} represents C{10}
        seconds.
        """
    self.assertEqual(10, dns.str2time('10S'))
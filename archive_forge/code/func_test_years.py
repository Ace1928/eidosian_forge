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
def test_years(self):
    """
        Like L{test_seconds}, but for the C{"Y"} suffix which multiplies the
        time value by C{31536000}, the number of seconds in a year.
        """
    self.assertEqual(6 * 31536000, dns.str2time('6Y'))
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
def test_unicodeName(self):
    """
        L{dns.Name} automatically encodes unicode domain name using C{idna}
        encoding.
        """
    name = dns.Name('échec.example.org')
    self.assertIsInstance(name.name, bytes)
    self.assertEqual(b'xn--chec-9oa.example.org', name.name)
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
class EDNSMessageStandardEncodingTests(StandardEncodingTestsMixin, unittest.SynchronousTestCase):
    """
    Tests for the encoding and decoding of various standard (non-EDNS) messages
    by L{dns._EDNSMessage}.
    """
    messageFactory = dns._EDNSMessage
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
def test_reprDefaults(self):
    """
        L{dns._EDNSMessage.__repr__} omits field values and sections which are
        identical to their defaults. The id field value is always shown.
        """
    self.assertEqual('<_EDNSMessage id=0>', repr(self.messageFactory()))
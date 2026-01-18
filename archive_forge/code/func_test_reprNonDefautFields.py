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
def test_reprNonDefautFields(self):
    """
        L{dns._EDNSMessage.__repr__} displays field values if they differ from
        their defaults.
        """
    m = self.messageFactory(id=10, opCode=20, rCode=30, maxSize=40, ednsVersion=50)
    self.assertEqual('<_EDNSMessage id=10 opCode=20 rCode=30 maxSize=40 ednsVersion=50>', repr(m))
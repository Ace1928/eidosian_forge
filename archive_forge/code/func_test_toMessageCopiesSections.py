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
def test_toMessageCopiesSections(self):
    """
        L{dns._EDNSMessage.toStr} makes no in place changes to the message
        instance.
        """
    ednsMessage = dns._EDNSMessage(ednsVersion=1)
    ednsMessage.toStr()
    self.assertEqual(ednsMessage.additional, [])
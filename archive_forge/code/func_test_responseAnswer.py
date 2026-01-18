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
def test_responseAnswer(self):
    """
        L{dns._responseFromMessage} sets the C{answer} flag to L{True}
        """
    request = dns.Message()
    response = dns._responseFromMessage(responseConstructor=dns.Message, message=request)
    self.assertEqual((False, True), (request.answer, response.answer))
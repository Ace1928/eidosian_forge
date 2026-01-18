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
def test_responseFromMessageResponseType(self):
    """
        L{dns.Message._responseFromMessage} is a constructor function which
        generates a new I{answer} message from an existing L{dns.Message} like
        instance.
        """
    request = dns.Message()
    response = dns._responseFromMessage(responseConstructor=dns.Message, message=request)
    self.assertIsNot(request, response)
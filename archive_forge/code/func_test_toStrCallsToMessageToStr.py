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
def test_toStrCallsToMessageToStr(self):
    """
        L{dns._EDNSMessage.toStr} calls C{toStr} on the message returned by
        L{dns._EDNSMessage._toMessage}.
        """
    m = dns._EDNSMessage()
    dummyBytes = object()

    class FakeMessage:
        """
            Fake Message
            """

        def toStr(self):
            """
                Fake toStr which returns dummyBytes.

                @return: dummyBytes
                """
            return dummyBytes

    def fakeToMessage(*args, **kwargs):
        return FakeMessage()
    m._toMessage = fakeToMessage
    self.assertEqual(dummyBytes, m.toStr())
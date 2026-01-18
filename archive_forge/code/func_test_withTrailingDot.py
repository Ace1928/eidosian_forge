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
def test_withTrailingDot(self):
    """
        L{dns._nameToLabels} returns a list ending with an empty
        label for a name with a trailing dot.
        """
    self.assertEqual(dns._nameToLabels(b'com.'), [b'com', b''])
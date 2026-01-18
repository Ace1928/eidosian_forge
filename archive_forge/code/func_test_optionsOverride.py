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
def test_optionsOverride(self):
    """
        L{dns._OPTHeader.options} can be overridden in the
        constructor.
        """
    h = dns._OPTHeader(options=[(1, 1, b'\x00')])
    self.assertEqual(h.options, [(1, 1, b'\x00')])
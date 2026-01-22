import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class UTCTests(TestCase):
    """
    Tests for L{amp.utc}.
    """

    def test_tzname(self):
        """
        L{amp.utc.tzname} returns C{"+00:00"}.
        """
        self.assertEqual(amp.utc.tzname(None), '+00:00')

    def test_dst(self):
        """
        L{amp.utc.dst} returns a zero timedelta.
        """
        self.assertEqual(amp.utc.dst(None), datetime.timedelta(0))

    def test_utcoffset(self):
        """
        L{amp.utc.utcoffset} returns a zero timedelta.
        """
        self.assertEqual(amp.utc.utcoffset(None), datetime.timedelta(0))

    def test_badSign(self):
        """
        L{amp._FixedOffsetTZInfo.fromSignHoursMinutes} raises L{ValueError} if
        passed an offset sign other than C{'+'} or C{'-'}.
        """
        self.assertRaises(ValueError, tz, '?', 0, 0)
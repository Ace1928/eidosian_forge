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
class Str2TimeTests(unittest.TestCase):
    """
    Tests for L{dns.str2name}.
    """

    def test_nonString(self):
        """
        When passed a non-string object, L{dns.str2name} returns it unmodified.
        """
        time = object()
        self.assertIs(time, dns.str2time(time))

    def test_seconds(self):
        """
        Passed a string giving a number of seconds, L{dns.str2time} returns the
        number of seconds represented.  For example, C{"10S"} represents C{10}
        seconds.
        """
        self.assertEqual(10, dns.str2time('10S'))

    def test_minutes(self):
        """
        Like C{test_seconds}, but for the C{"M"} suffix which multiplies the
        time value by C{60} (the number of seconds in a minute!).
        """
        self.assertEqual(2 * 60, dns.str2time('2M'))

    def test_hours(self):
        """
        Like C{test_seconds}, but for the C{"H"} suffix which multiplies the
        time value by C{3600}, the number of seconds in an hour.
        """
        self.assertEqual(3 * 3600, dns.str2time('3H'))

    def test_days(self):
        """
        Like L{test_seconds}, but for the C{"D"} suffix which multiplies the
        time value by C{86400}, the number of seconds in a day.
        """
        self.assertEqual(4 * 86400, dns.str2time('4D'))

    def test_weeks(self):
        """
        Like L{test_seconds}, but for the C{"W"} suffix which multiplies the
        time value by C{604800}, the number of seconds in a week.
        """
        self.assertEqual(5 * 604800, dns.str2time('5W'))

    def test_years(self):
        """
        Like L{test_seconds}, but for the C{"Y"} suffix which multiplies the
        time value by C{31536000}, the number of seconds in a year.
        """
        self.assertEqual(6 * 31536000, dns.str2time('6Y'))

    def test_invalidPrefix(self):
        """
        If a non-integer prefix is given, L{dns.str2time} raises L{ValueError}.
        """
        self.assertRaises(ValueError, dns.str2time, 'fooS')

    def test_invalidSuffix(self) -> None:
        """
        If an invalid suffix is given, L{dns.str2time} raises L{ValueError}.
        """
        self.assertRaises(ValueError, dns.str2time, '1Q')
from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
class TimeFormattingTests(unittest.TestCase):
    """
    Tests for time formatting functions.
    """

    def setUp(self) -> None:
        addTZCleanup(self)

    def test_formatTimeWithDefaultFormat(self) -> None:
        """
        Default time stamp format is RFC 3339 and offset respects the timezone
        as set by the standard C{TZ} environment variable and L{tzset} API.
        """
        if tzset is None:
            raise SkipTest('Platform cannot change timezone; unable to verify offsets.')

        def testForTimeZone(name: str, expectedDST: Optional[str], expectedSTD: str) -> None:
            setTZ(name)
            localSTD = mktime((2007, 1, 31, 0, 0, 0, 2, 31, 0))
            self.assertEqual(formatTime(localSTD), expectedSTD)
            if expectedDST:
                localDST = mktime((2006, 6, 30, 0, 0, 0, 4, 181, 1))
                self.assertEqual(formatTime(localDST), expectedDST)
        testForTimeZone('UTC+00', None, '2007-01-31T00:00:00+0000')
        testForTimeZone('EST+05EDT,M4.1.0,M10.5.0', '2006-06-30T00:00:00-0400', '2007-01-31T00:00:00-0500')
        testForTimeZone('CEST-01CEDT,M4.1.0,M10.5.0', '2006-06-30T00:00:00+0200', '2007-01-31T00:00:00+0100')
        testForTimeZone('CST+06', None, '2007-01-31T00:00:00-0600')

    def test_formatTimeWithNoTime(self) -> None:
        """
        If C{when} argument is L{None}, we get the default output.
        """
        self.assertEqual(formatTime(None), '-')
        self.assertEqual(formatTime(None, default='!'), '!')

    def test_formatTimeWithNoFormat(self) -> None:
        """
        If C{timeFormat} argument is L{None}, we get the default output.
        """
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        self.assertEqual(formatTime(t, timeFormat=None), '-')
        self.assertEqual(formatTime(t, timeFormat=None, default='!'), '!')

    def test_formatTimeWithAlternateTimeFormat(self) -> None:
        """
        Alternate time format in output.
        """
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        self.assertEqual(formatTime(t, timeFormat='%Y/%W'), '2013/38')

    def test_formatTimePercentF(self) -> None:
        """
        "%f" supported in time format.
        """
        self.assertEqual(formatTime(1000000.23456, timeFormat='%f'), '234560')
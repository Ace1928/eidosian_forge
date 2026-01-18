import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_toRFC4034DateString(self):
    """
        L{DateSerialNumber.toRFC4034DateString} interprets the current value as
        a unix timestamp and returns a date string representation of that date.
        """
    self.assertEqual('20120101000000', SerialNumber(1325376000).toRFC4034DateString())
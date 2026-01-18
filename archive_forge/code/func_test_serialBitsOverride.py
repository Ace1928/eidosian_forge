import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_serialBitsOverride(self):
    """
        L{SerialNumber.__init__} accepts a C{serialBits} argument whose value is
        assigned to L{SerialNumber.serialBits}.
        """
    self.assertEqual(SerialNumber(1, serialBits=8)._serialBits, 8)
import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
def test_surprisingAddition(self):
    """
        Note that 100+100 > 100, but that (100+100)+100 < 100.  Incrementing a
        serial number can cause it to become "smaller".  Of course, incrementing
        by a smaller number will allow many more increments to be made before
        this occurs.  However this is always something to be aware of, it can
        cause surprising errors, or be useful as it is the only defined way to
        actually cause a serial number to decrease.
        """
    self.assertTrue(serialNumber8(100) + serialNumber8(100) > serialNumber8(100))
    self.assertTrue(serialNumber8(100) + serialNumber8(100) + serialNumber8(100) < serialNumber8(100))
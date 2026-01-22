import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
class SerialNumber2BitTests(unittest.TestCase):
    """
    Tests for correct answers to example calculations in RFC1982 5.1.

    The simplest meaningful serial number space has SERIAL_BITS == 2.  In this
    space, the integers that make up the serial number space are 0, 1, 2, and 3.
    That is, 3 == 2^SERIAL_BITS - 1.

    https://tools.ietf.org/html/rfc1982#section-5.1
    """

    def test_maxadd(self):
        """
        In this space, the largest integer that it is meaningful to add to a
        sequence number is 2^(SERIAL_BITS - 1) - 1, or 1.
        """
        self.assertEqual(SerialNumber(0, serialBits=2)._maxAdd, 1)

    def test_add(self):
        """
        Then, as defined 0+1 == 1, 1+1 == 2, 2+1 == 3, and 3+1 == 0.
        """
        self.assertEqual(serialNumber2(0) + serialNumber2(1), serialNumber2(1))
        self.assertEqual(serialNumber2(1) + serialNumber2(1), serialNumber2(2))
        self.assertEqual(serialNumber2(2) + serialNumber2(1), serialNumber2(3))
        self.assertEqual(serialNumber2(3) + serialNumber2(1), serialNumber2(0))

    def test_gt(self):
        """
        Further, 1 > 0, 2 > 1, 3 > 2, and 0 > 3.
        """
        self.assertTrue(serialNumber2(1) > serialNumber2(0))
        self.assertTrue(serialNumber2(2) > serialNumber2(1))
        self.assertTrue(serialNumber2(3) > serialNumber2(2))
        self.assertTrue(serialNumber2(0) > serialNumber2(3))

    def test_undefined(self):
        """
        It is undefined whether 2 > 0 or 0 > 2, and whether 1 > 3 or 3 > 1.
        """
        assertUndefinedComparison(self, serialNumber2(2), serialNumber2(0))
        assertUndefinedComparison(self, serialNumber2(0), serialNumber2(2))
        assertUndefinedComparison(self, serialNumber2(1), serialNumber2(3))
        assertUndefinedComparison(self, serialNumber2(3), serialNumber2(1))
import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest
class SerialNumberTests(unittest.TestCase):
    """
    Tests for L{SerialNumber}.
    """

    def test_serialBitsDefault(self):
        """
        L{SerialNumber.serialBits} has default value 32.
        """
        self.assertEqual(SerialNumber(1)._serialBits, 32)

    def test_serialBitsOverride(self):
        """
        L{SerialNumber.__init__} accepts a C{serialBits} argument whose value is
        assigned to L{SerialNumber.serialBits}.
        """
        self.assertEqual(SerialNumber(1, serialBits=8)._serialBits, 8)

    def test_repr(self):
        """
        L{SerialNumber.__repr__} returns a string containing number and
        serialBits.
        """
        self.assertEqual('<SerialNumber number=123 serialBits=32>', repr(SerialNumber(123, serialBits=32)))

    def test_str(self):
        """
        L{SerialNumber.__str__} returns a string representation of the current
        value.
        """
        self.assertEqual(str(SerialNumber(123)), '123')

    def test_int(self):
        """
        L{SerialNumber.__int__} returns an integer representation of the current
        value.
        """
        self.assertEqual(int(SerialNumber(123)), 123)

    def test_hash(self):
        """
        L{SerialNumber.__hash__} allows L{SerialNumber} instances to be hashed
        for use as dictionary keys.
        """
        self.assertEqual(hash(SerialNumber(1)), hash(SerialNumber(1)))
        self.assertNotEqual(hash(SerialNumber(1)), hash(SerialNumber(2)))

    def test_convertOtherSerialBitsMismatch(self):
        """
        L{SerialNumber._convertOther} raises L{TypeError} if the other
        SerialNumber instance has a different C{serialBits} value.
        """
        s1 = SerialNumber(0, serialBits=8)
        s2 = SerialNumber(0, serialBits=16)
        self.assertRaises(TypeError, s1._convertOther, s2)

    def test_eq(self):
        """
        L{SerialNumber.__eq__} provides rich equality comparison.
        """
        self.assertEqual(SerialNumber(1), SerialNumber(1))

    def test_eqForeignType(self):
        """
        == comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        returns L{NotImplemented}.
        """
        self.assertFalse(SerialNumber(1) == object())
        self.assertIs(SerialNumber(1).__eq__(object()), NotImplemented)

    def test_ne(self):
        """
        L{SerialNumber.__ne__} provides rich equality comparison.
        """
        self.assertFalse(SerialNumber(1) != SerialNumber(1))
        self.assertNotEqual(SerialNumber(1), SerialNumber(2))

    def test_neForeignType(self):
        """
        != comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        returns L{NotImplemented}.
        """
        self.assertTrue(SerialNumber(1) != object())
        self.assertIs(SerialNumber(1).__ne__(object()), NotImplemented)

    def test_le(self):
        """
        L{SerialNumber.__le__} provides rich <= comparison.
        """
        self.assertTrue(SerialNumber(1) <= SerialNumber(1))
        self.assertTrue(SerialNumber(1) <= SerialNumber(2))

    def test_leForeignType(self):
        """
        <= comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        raises L{TypeError}.
        """
        self.assertRaises(TypeError, lambda: SerialNumber(1) <= object())

    def test_ge(self):
        """
        L{SerialNumber.__ge__} provides rich >= comparison.
        """
        self.assertTrue(SerialNumber(1) >= SerialNumber(1))
        self.assertTrue(SerialNumber(2) >= SerialNumber(1))

    def test_geForeignType(self):
        """
        >= comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        raises L{TypeError}.
        """
        self.assertRaises(TypeError, lambda: SerialNumber(1) >= object())

    def test_lt(self):
        """
        L{SerialNumber.__lt__} provides rich < comparison.
        """
        self.assertTrue(SerialNumber(1) < SerialNumber(2))

    def test_ltForeignType(self):
        """
        < comparison of L{SerialNumber} with a non-L{SerialNumber} instance
        raises L{TypeError}.
        """
        self.assertRaises(TypeError, lambda: SerialNumber(1) < object())

    def test_gt(self):
        """
        L{SerialNumber.__gt__} provides rich > comparison.
        """
        self.assertTrue(SerialNumber(2) > SerialNumber(1))

    def test_gtForeignType(self):
        """
        > comparison of L{SerialNumber} with a non-L{SerialNumber} instance
          raises L{TypeError}.
        """
        self.assertRaises(TypeError, lambda: SerialNumber(2) > object())

    def test_add(self):
        """
        L{SerialNumber.__add__} allows L{SerialNumber} instances to be summed.
        """
        self.assertEqual(SerialNumber(1) + SerialNumber(1), SerialNumber(2))

    def test_addForeignType(self):
        """
        Addition of L{SerialNumber} with a non-L{SerialNumber} instance raises
        L{TypeError}.
        """
        self.assertRaises(TypeError, lambda: SerialNumber(1) + object())

    def test_addOutOfRangeHigh(self):
        """
        L{SerialNumber} cannot be added with other SerialNumber values larger
        than C{_maxAdd}.
        """
        maxAdd = SerialNumber(1)._maxAdd
        self.assertRaises(ArithmeticError, lambda: SerialNumber(1) + SerialNumber(maxAdd + 1))

    def test_maxVal(self):
        """
        L{SerialNumber.__add__} returns a wrapped value when s1 plus the s2
        would result in a value greater than the C{maxVal}.
        """
        s = SerialNumber(1)
        maxVal = s._halfRing + s._halfRing - 1
        maxValPlus1 = maxVal + 1
        self.assertTrue(SerialNumber(maxValPlus1) > SerialNumber(maxVal))
        self.assertEqual(SerialNumber(maxValPlus1), SerialNumber(0))

    def test_fromRFC4034DateString(self):
        """
        L{SerialNumber.fromRFC4034DateString} accepts a datetime string argument
        of the form 'YYYYMMDDhhmmss' and returns an L{SerialNumber} instance
        whose value is the unix timestamp corresponding to that UTC date.
        """
        self.assertEqual(SerialNumber(1325376000), SerialNumber.fromRFC4034DateString('20120101000000'))

    def test_toRFC4034DateString(self):
        """
        L{DateSerialNumber.toRFC4034DateString} interprets the current value as
        a unix timestamp and returns a date string representation of that date.
        """
        self.assertEqual('20120101000000', SerialNumber(1325376000).toRFC4034DateString())

    def test_unixEpoch(self):
        """
        L{SerialNumber.toRFC4034DateString} stores 32bit timestamps relative to
        the UNIX epoch.
        """
        self.assertEqual(SerialNumber(0).toRFC4034DateString(), '19700101000000')

    def test_Y2106Problem(self):
        """
        L{SerialNumber} wraps unix timestamps in the year 2106.
        """
        self.assertEqual(SerialNumber(-1).toRFC4034DateString(), '21060207062815')

    def test_Y2038Problem(self):
        """
        L{SerialNumber} raises ArithmeticError when used to add dates more than
        68 years in the future.
        """
        maxAddTime = calendar.timegm(datetime(2038, 1, 19, 3, 14, 7).utctimetuple())
        self.assertEqual(maxAddTime, SerialNumber(0)._maxAdd)
        self.assertRaises(ArithmeticError, lambda: SerialNumber(0) + SerialNumber(maxAddTime + 1))
import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
class TestPythonTimeBuilder_UtiltyFunctions(unittest.TestCase):

    def test_year_range_check(self):
        yearlimit = Limit('Invalid year string.', 0, 9999, YearOutOfBoundsError, 'Year must be between 1..9999.', None)
        self.assertEqual(year_range_check('1', yearlimit), 1000)

    def test_fractional_range_check(self):
        limit = Limit('Invalid string.', -1, 1, ValueError, 'Value must be between -1..1.', None)
        self.assertEqual(fractional_range_check(10, '1', limit), 1)
        self.assertEqual(fractional_range_check(10, '-1', limit), -1)
        self.assertEqual(fractional_range_check(10, '0.1', limit), FractionalComponent(0, 1))
        self.assertEqual(fractional_range_check(10, '-0.1', limit), FractionalComponent(-0, 1))
        with self.assertRaises(ValueError):
            fractional_range_check(10, '1.1', limit)
        with self.assertRaises(ValueError):
            fractional_range_check(10, '-1.1', limit)

    def test_cast_to_fractional_component(self):
        self.assertEqual(_cast_to_fractional_component(10, '1.1'), FractionalComponent(1, 1))
        self.assertEqual(_cast_to_fractional_component(10, '-1.1'), FractionalComponent(-1, 1))
        self.assertEqual(_cast_to_fractional_component(100, '1.1'), FractionalComponent(1, 10))
        self.assertEqual(_cast_to_fractional_component(100, '-1.1'), FractionalComponent(-1, 10))
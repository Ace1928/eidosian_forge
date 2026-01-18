import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_year_range_check(self):
    yearlimit = Limit('Invalid year string.', 0, 9999, YearOutOfBoundsError, 'Year must be between 1..9999.', None)
    self.assertEqual(year_range_check('1', yearlimit), 1000)
import unittest
import operator
from datetime import timedelta, date, datetime
from isodate import Duration, parse_duration, ISO8601Error
from isodate import D_DEFAULT, D_WEEK, D_ALT_EXT, duration_isoformat
class TestDateCalc(unittest.TestCase):
    """
        A test case template test addition operators for Duration objects.
        """

    def test_calc(self):
        """
            Test operator +.
            """
        if expectation is None:
            self.assertRaises(ValueError, operator.add, start, duration)
        else:
            self.assertEqual(start + duration, expectation)
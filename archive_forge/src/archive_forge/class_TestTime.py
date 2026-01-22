import unittest
from datetime import time
from isodate import parse_time, UTC, FixedOffset, ISO8601Error, time_isoformat
from isodate import TIME_BAS_COMPLETE, TIME_BAS_MINUTE
from isodate import TIME_EXT_COMPLETE, TIME_EXT_MINUTE
from isodate import TIME_HOUR
from isodate import TZ_BAS, TZ_EXT, TZ_HOUR
class TestTime(unittest.TestCase):
    """
        A test case template to parse an ISO time string into a time
        object.
        """

    def test_parse(self):
        """
            Parse an ISO time string and compare it to the expected value.
            """
        if expectation is None:
            self.assertRaises(ISO8601Error, parse_time, timestring)
        else:
            result = parse_time(timestring)
            self.assertEqual(result, expectation)

    def test_format(self):
        """
            Take time object and create ISO string from it.
            This is the reverse test to test_parse.
            """
        if expectation is None:
            self.assertRaises(AttributeError, time_isoformat, expectation, format)
        elif format is not None:
            self.assertEqual(time_isoformat(expectation, format), timestring)
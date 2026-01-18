import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_repeating_interval_badstr(self):
    testtuples = ('bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_repeating_interval(testtuple, builder=None)
import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_repeating_interval_suffixgarbage(self):
    with self.assertRaises(ISOFormatError):
        parse_repeating_interval('R3/1981-04-05/P1Dasdf', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_repeating_interval('R3/1981-04-05/P0003-06-04T12:30:05.5asdfasdf', builder=None)
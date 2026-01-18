import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_interval_badtype(self):
    testtuples = (None, 1, False, 1.234)
    for testtuple in testtuples:
        with self.assertRaises(ValueError):
            parse_interval(testtuple, builder=None)
import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_time_badtype(self):
    testtuples = (None, 1, False, 1.234)
    for testtuple in testtuples:
        with self.assertRaises(ValueError):
            parse_time(testtuple, builder=None)
import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_time_badstr(self):
    testtuples = ('A6:14:00.000123Z', '06:14:0B', '06:1 :02', '0000,70:24,9', '00.27:5332', 'bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_time(testtuple, builder=None)
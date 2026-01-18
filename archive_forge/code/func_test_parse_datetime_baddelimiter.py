import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime_baddelimiter(self):
    testtuples = ('1981-04-05,23:21:28,512400Z', '2004-W53-6 23:21:28.512400-12:3', '1981040523:21:28')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_datetime(testtuple, builder=None)
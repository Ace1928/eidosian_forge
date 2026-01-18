import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime_mockbuilder(self):
    mockBuilder = mock.Mock()
    expectedargs = (DateTuple('1981', None, None, None, None, '095'), TimeTuple('23', '21', '28.512400', TimezoneTuple(True, None, '12', '34', '-12:34')))
    mockBuilder.build_datetime.return_value = expectedargs
    result = parse_datetime('1981095T23:21:28.512400-12:34', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_datetime.assert_called_once_with(*expectedargs)
import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_time_mockbuilder(self):
    mockBuilder = mock.Mock()
    expectedargs = {'hh': '01', 'mm': '23', 'ss': '45', 'tz': None}
    mockBuilder.build_time.return_value = expectedargs
    result = parse_time('01:23:45', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_time.assert_called_once_with(**expectedargs)
    mockBuilder = mock.Mock()
    expectedargs = {'hh': '23', 'mm': '21', 'ss': '28.512400', 'tz': TimezoneTuple(False, None, '00', '00', '+00:00')}
    mockBuilder.build_time.return_value = expectedargs
    result = parse_time('232128.512400+00:00', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_time.assert_called_once_with(**expectedargs)
    mockBuilder = mock.Mock()
    expectedargs = {'hh': '23', 'mm': '21', 'ss': '28.512400', 'tz': TimezoneTuple(False, None, '11', '15', '+11:15')}
    mockBuilder.build_time.return_value = expectedargs
    result = parse_time('23:21:28.512400+11:15', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_time.assert_called_once_with(**expectedargs)
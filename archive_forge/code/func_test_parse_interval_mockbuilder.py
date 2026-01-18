import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_parse_interval_mockbuilder(self):
    mockBuilder = mock.Mock()
    expectedargs = {'end': DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None)), 'duration': DurationTuple(None, '1', None, None, None, None, None)}
    mockBuilder.build_interval.return_value = expectedargs
    result = parse_interval('P1M/1981-04-05T01:01:00', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_interval.assert_called_once_with(**expectedargs)
    mockBuilder = mock.Mock()
    expectedargs = {'start': DateTuple('2014', '11', '12', None, None, None), 'duration': DurationTuple(None, None, None, None, '1', None, None)}
    mockBuilder.build_interval.return_value = expectedargs
    result = parse_interval('2014-11-12/PT1H', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_interval.assert_called_once_with(**expectedargs)
    mockBuilder = mock.Mock()
    expectedargs = {'start': DatetimeTuple(DateTuple('1980', '03', '05', None, None, None), TimeTuple('01', '01', '00', None)), 'end': DatetimeTuple(DateTuple('1981', '04', '05', None, None, None), TimeTuple('01', '01', '00', None))}
    mockBuilder.build_interval.return_value = expectedargs
    result = parse_interval('1980-03-05T01:01:00/1981-04-05T01:01:00', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_interval.assert_called_once_with(**expectedargs)
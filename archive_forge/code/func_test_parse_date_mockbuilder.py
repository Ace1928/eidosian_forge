import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_parse_date_mockbuilder(self):
    mockBuilder = mock.Mock()
    expectedargs = {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}
    mockBuilder.build_date.return_value = expectedargs
    result = parse_date('1981-04-05', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_date.assert_called_once_with(**expectedargs)
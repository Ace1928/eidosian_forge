import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime_spacedelimited(self):
    expectedargs = (DateTuple('2004', None, None, '53', '6', None), TimeTuple('23', '21', '28.512400', TimezoneTuple(True, None, '12', '34', '-12:34')))
    with mock.patch.object(aniso8601.time.PythonTimeBuilder, 'build_datetime') as mockBuildDateTime:
        mockBuildDateTime.return_value = expectedargs
        result = parse_datetime('2004-W53-6 23:21:28.512400-12:34', delimiter=' ')
    self.assertEqual(result, expectedargs)
    mockBuildDateTime.assert_called_once_with(*expectedargs)
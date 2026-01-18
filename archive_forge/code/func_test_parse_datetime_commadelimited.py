import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime_commadelimited(self):
    expectedargs = (DateTuple('1981', '04', '05', None, None, None), TimeTuple('23', '21', '28.512400', TimezoneTuple(False, True, None, None, 'Z')))
    with mock.patch.object(aniso8601.time.PythonTimeBuilder, 'build_datetime') as mockBuildDateTime:
        mockBuildDateTime.return_value = expectedargs
        result = parse_datetime('1981-04-05,23:21:28,512400Z', delimiter=',')
    self.assertEqual(result, expectedargs)
    mockBuildDateTime.assert_called_once_with(*expectedargs)
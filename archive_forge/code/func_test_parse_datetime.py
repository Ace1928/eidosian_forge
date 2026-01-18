import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_parse_datetime(self):
    testtuples = (('2019-06-05T01:03:11,858714', (DateTuple('2019', '06', '05', None, None, None), TimeTuple('01', '03', '11.858714', None))), ('2019-06-05T01:03:11.858714', (DateTuple('2019', '06', '05', None, None, None), TimeTuple('01', '03', '11.858714', None))), ('1981-04-05T23:21:28.512400Z', (DateTuple('1981', '04', '05', None, None, None), TimeTuple('23', '21', '28.512400', TimezoneTuple(False, True, None, None, 'Z')))), ('1981095T23:21:28.512400-12:34', (DateTuple('1981', None, None, None, None, '095'), TimeTuple('23', '21', '28.512400', TimezoneTuple(True, None, '12', '34', '-12:34')))), ('19810405T23:21:28+00', (DateTuple('1981', '04', '05', None, None, None), TimeTuple('23', '21', '28', TimezoneTuple(False, None, '00', None, '+00')))), ('19810405T23:21:28+00:00', (DateTuple('1981', '04', '05', None, None, None), TimeTuple('23', '21', '28', TimezoneTuple(False, None, '00', '00', '+00:00')))))
    for testtuple in testtuples:
        with mock.patch.object(aniso8601.time.PythonTimeBuilder, 'build_datetime') as mockBuildDateTime:
            mockBuildDateTime.return_value = testtuple[1]
            result = parse_datetime(testtuple[0])
        self.assertEqual(result, testtuple[1])
        mockBuildDateTime.assert_called_once_with(*testtuple[1])
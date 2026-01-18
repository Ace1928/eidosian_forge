import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_parse_date(self):
    testtuples = (('2013', {'YYYY': '2013', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('0001', {'YYYY': '0001', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('19', {'YYYY': '19', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('1981-04-05', {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}), ('19810405', {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}), ('1981-04', {'YYYY': '1981', 'MM': '04', 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('2004-W53', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': None, 'DDD': None}), ('2009-W01', {'YYYY': '2009', 'MM': None, 'DD': None, 'Www': '01', 'D': None, 'DDD': None}), ('2004-W53-6', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': '6', 'DDD': None}), ('2004W53', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': None, 'DDD': None}), ('2004W536', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': '6', 'DDD': None}), ('1981-095', {'YYYY': '1981', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '095'}), ('1981095', {'YYYY': '1981', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '095'}), ('1980366', {'YYYY': '1980', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '366'}))
    for testtuple in testtuples:
        with mock.patch.object(aniso8601.date.PythonTimeBuilder, 'build_date') as mockBuildDate:
            mockBuildDate.return_value = testtuple[1]
            result = parse_date(testtuple[0])
            self.assertEqual(result, testtuple[1])
            mockBuildDate.assert_called_once_with(**testtuple[1])
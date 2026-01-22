import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
class TestDateParserFunctions(unittest.TestCase):

    def test_parse_date(self):
        testtuples = (('2013', {'YYYY': '2013', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('0001', {'YYYY': '0001', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('19', {'YYYY': '19', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('1981-04-05', {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}), ('19810405', {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}), ('1981-04', {'YYYY': '1981', 'MM': '04', 'DD': None, 'Www': None, 'D': None, 'DDD': None}), ('2004-W53', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': None, 'DDD': None}), ('2009-W01', {'YYYY': '2009', 'MM': None, 'DD': None, 'Www': '01', 'D': None, 'DDD': None}), ('2004-W53-6', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': '6', 'DDD': None}), ('2004W53', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': None, 'DDD': None}), ('2004W536', {'YYYY': '2004', 'MM': None, 'DD': None, 'Www': '53', 'D': '6', 'DDD': None}), ('1981-095', {'YYYY': '1981', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '095'}), ('1981095', {'YYYY': '1981', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '095'}), ('1980366', {'YYYY': '1980', 'MM': None, 'DD': None, 'Www': None, 'D': None, 'DDD': '366'}))
        for testtuple in testtuples:
            with mock.patch.object(aniso8601.date.PythonTimeBuilder, 'build_date') as mockBuildDate:
                mockBuildDate.return_value = testtuple[1]
                result = parse_date(testtuple[0])
                self.assertEqual(result, testtuple[1])
                mockBuildDate.assert_called_once_with(**testtuple[1])

    def test_parse_date_badtype(self):
        testtuples = (None, 1, False, 1.234)
        for testtuple in testtuples:
            with self.assertRaises(ValueError):
                parse_date(testtuple, builder=None)

    def test_parse_date_badstr(self):
        testtuples = ('W53', '2004-W', '2014-01-230', '2014-012-23', '201-01-23', '201401230', '201401', '9999 W53', '20.50230', '198104', 'bad', '')
        for testtuple in testtuples:
            with self.assertRaises(ISOFormatError):
                parse_date(testtuple, builder=None)

    def test_parse_date_mockbuilder(self):
        mockBuilder = mock.Mock()
        expectedargs = {'YYYY': '1981', 'MM': '04', 'DD': '05', 'Www': None, 'D': None, 'DDD': None}
        mockBuilder.build_date.return_value = expectedargs
        result = parse_date('1981-04-05', builder=mockBuilder)
        self.assertEqual(result, expectedargs)
        mockBuilder.build_date.assert_called_once_with(**expectedargs)
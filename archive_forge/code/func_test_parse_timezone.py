import unittest
import aniso8601
from aniso8601.exceptions import ISOFormatError
from aniso8601.tests.compat import mock
from aniso8601.timezone import parse_timezone
def test_parse_timezone(self):
    testtuples = (('Z', {'negative': False, 'Z': True, 'name': 'Z'}), ('+00:00', {'negative': False, 'hh': '00', 'mm': '00', 'name': '+00:00'}), ('+01:00', {'negative': False, 'hh': '01', 'mm': '00', 'name': '+01:00'}), ('-01:00', {'negative': True, 'hh': '01', 'mm': '00', 'name': '-01:00'}), ('+00:12', {'negative': False, 'hh': '00', 'mm': '12', 'name': '+00:12'}), ('+01:23', {'negative': False, 'hh': '01', 'mm': '23', 'name': '+01:23'}), ('-01:23', {'negative': True, 'hh': '01', 'mm': '23', 'name': '-01:23'}), ('+0000', {'negative': False, 'hh': '00', 'mm': '00', 'name': '+0000'}), ('+0100', {'negative': False, 'hh': '01', 'mm': '00', 'name': '+0100'}), ('-0100', {'negative': True, 'hh': '01', 'mm': '00', 'name': '-0100'}), ('+0012', {'negative': False, 'hh': '00', 'mm': '12', 'name': '+0012'}), ('+0123', {'negative': False, 'hh': '01', 'mm': '23', 'name': '+0123'}), ('-0123', {'negative': True, 'hh': '01', 'mm': '23', 'name': '-0123'}), ('+00', {'negative': False, 'hh': '00', 'mm': None, 'name': '+00'}), ('+01', {'negative': False, 'hh': '01', 'mm': None, 'name': '+01'}), ('-01', {'negative': True, 'hh': '01', 'mm': None, 'name': '-01'}), ('+12', {'negative': False, 'hh': '12', 'mm': None, 'name': '+12'}), ('-12', {'negative': True, 'hh': '12', 'mm': None, 'name': '-12'}))
    for testtuple in testtuples:
        with mock.patch.object(aniso8601.timezone.PythonTimeBuilder, 'build_timezone') as mockBuildTimezone:
            mockBuildTimezone.return_value = testtuple[1]
            result = parse_timezone(testtuple[0])
            self.assertEqual(result, testtuple[1])
            mockBuildTimezone.assert_called_once_with(**testtuple[1])
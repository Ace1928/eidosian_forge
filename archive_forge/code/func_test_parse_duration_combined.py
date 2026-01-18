import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_combined(self):
    testtuples = (('P0003-06-04T12:30:05', {'PnY': '0003', 'PnM': '06', 'PnD': '04', 'TnH': '12', 'TnM': '30', 'TnS': '05'}), ('P0003-06-04T12:30:05,5', {'PnY': '0003', 'PnM': '06', 'PnD': '04', 'TnH': '12', 'TnM': '30', 'TnS': '05.5'}), ('P0003-06-04T12:30:05.5', {'PnY': '0003', 'PnM': '06', 'PnD': '04', 'TnH': '12', 'TnM': '30', 'TnS': '05.5'}), ('P0001-02-03T14:43:59.9999997', {'PnY': '0001', 'PnM': '02', 'PnD': '03', 'TnH': '14', 'TnM': '43', 'TnS': '59.9999997'}))
    for testtuple in testtuples:
        result = _parse_duration_combined(testtuple[0])
        self.assertEqual(result, testtuple[1])
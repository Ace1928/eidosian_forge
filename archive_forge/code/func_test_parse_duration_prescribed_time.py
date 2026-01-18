import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed_time(self):
    testtuples = (('P1Y2M3DT4H54M6S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6'}), ('P1Y2M3DT4H54M6,5S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('P1Y2M3DT4H54M6.5S', {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('PT4H54M6,5S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('PT4H54M6.5S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '4', 'TnM': '54', 'TnS': '6.5'}), ('PT4H', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '4', 'TnM': None, 'TnS': None}), ('PT5M', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': None, 'TnM': '5', 'TnS': None}), ('PT6S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': None, 'TnM': None, 'TnS': '6'}), ('PT1H2M', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '1', 'TnM': '2', 'TnS': None}), ('PT3H4S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': '3', 'TnM': None, 'TnS': '4'}), ('PT5M6S', {'PnY': None, 'PnM': None, 'PnW': None, 'PnD': None, 'TnH': None, 'TnM': '5', 'TnS': '6'}))
    for testtuple in testtuples:
        result = _parse_duration_prescribed_time(testtuple[0])
        self.assertEqual(result, testtuple[1])
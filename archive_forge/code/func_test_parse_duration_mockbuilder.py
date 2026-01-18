import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_mockbuilder(self):
    mockBuilder = mock.Mock()
    expectedargs = {'PnY': '1', 'PnM': '2', 'PnW': None, 'PnD': '3', 'TnH': '4', 'TnM': '54', 'TnS': '6'}
    mockBuilder.build_duration.return_value = expectedargs
    result = parse_duration('P1Y2M3DT4H54M6S', builder=mockBuilder)
    self.assertEqual(result, expectedargs)
    mockBuilder.build_duration.assert_called_once_with(**expectedargs)
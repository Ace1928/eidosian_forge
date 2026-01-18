import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed_time_outoforder(self):
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('1Y2M3D1SPT1M')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('P1Y2M3D2MT1S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('P2M3D1ST1Y1M')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('P1Y2M2MT3D1S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('PT1S1H')
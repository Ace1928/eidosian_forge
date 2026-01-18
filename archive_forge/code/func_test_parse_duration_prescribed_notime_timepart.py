import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed_notime_timepart(self):
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_notime('P1S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_notime('P1D1S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_notime('P1H1M')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_notime('P1Y2M3D4H')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_notime('P1Y2M3D4H5S')
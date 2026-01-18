import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed_negative(self):
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-1Y')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-2M')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-3D')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-4W')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-1Y2M3D')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-T1H')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-T2M')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-T3S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed('P-1Y2M3DT4H54M6S')
import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_outoforder(self):
    with self.assertRaises(ISOFormatError):
        parse_duration('P1S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1D1S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1H1M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('1Y2M3D1SPT1M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1Y2M3D2MT1S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P2M3D1ST1Y1M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1Y2M2MT3D1S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1D1Y1M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('PT1S1H', builder=None)
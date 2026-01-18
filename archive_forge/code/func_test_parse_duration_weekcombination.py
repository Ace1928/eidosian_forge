import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_weekcombination(self):
    with self.assertRaises(ISOFormatError):
        parse_duration('P1Y2W', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1M2W', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P2W3D', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1Y2W3D', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1M2W3D', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P1Y1M2W3D', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT4H', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT54M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT6S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT4H54M', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT4H6S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT54M6S', builder=None)
    with self.assertRaises(ISOFormatError):
        parse_duration('P7WT4H54M6S', builder=None)
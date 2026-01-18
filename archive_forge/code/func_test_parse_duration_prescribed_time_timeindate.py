import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_prescribed_time_timeindate(self):
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('P1Y2M3D4HT54M6S')
    with self.assertRaises(ISOFormatError):
        _parse_duration_prescribed_time('P1Y2M3D6ST4H54M')
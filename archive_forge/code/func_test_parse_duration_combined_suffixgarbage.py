import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_combined_suffixgarbage(self):
    with self.assertRaises(ISOFormatError):
        _parse_duration_combined('P0003-06-04T12:30:05.5asdfasdf')
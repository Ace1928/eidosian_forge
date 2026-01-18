import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_parse_duration_badtype(self):
    testtuples = (None, 1, False, 1.234)
    for testtuple in testtuples:
        with self.assertRaises(ValueError):
            parse_duration(testtuple, builder=None)
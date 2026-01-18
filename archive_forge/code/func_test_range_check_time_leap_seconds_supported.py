import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
def test_range_check_time_leap_seconds_supported(self):
    self.assertEqual(LeapSecondSupportingTestBuilder.range_check_time(hh='23', mm='59', ss='60'), (23, 59, 60, None))
    with self.assertRaises(SecondsOutOfBoundsError):
        LeapSecondSupportingTestBuilder.range_check_time(hh='01', mm='02', ss='60')
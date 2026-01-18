from unittest import TestCase
from fastimport import (
def test_parse_tz_utc(self):
    self.assertEqual(0, dates.parse_tz(b'+0000'))
    self.assertEqual(0, dates.parse_tz(b'-0000'))
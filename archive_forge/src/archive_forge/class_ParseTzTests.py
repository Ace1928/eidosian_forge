from unittest import TestCase
from fastimport import (
class ParseTzTests(TestCase):

    def test_parse_tz_utc(self):
        self.assertEqual(0, dates.parse_tz(b'+0000'))
        self.assertEqual(0, dates.parse_tz(b'-0000'))

    def test_parse_tz_cet(self):
        self.assertEqual(3600, dates.parse_tz(b'+0100'))

    def test_parse_tz_odd(self):
        self.assertEqual(1864800, dates.parse_tz(b'+51800'))
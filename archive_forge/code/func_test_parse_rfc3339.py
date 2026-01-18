import unittest
from datetime import datetime
from .dateutil import UTC, TimezoneInfo, format_rfc3339, parse_rfc3339
def test_parse_rfc3339(self):
    self._parse_rfc3339_test('2017-07-25T04:44:21Z', 2017, 7, 25, 4, 44, 21)
    self._parse_rfc3339_test('2017-07-25 04:44:21Z', 2017, 7, 25, 4, 44, 21)
    self._parse_rfc3339_test('2017-07-25T04:44:21', 2017, 7, 25, 4, 44, 21)
    self._parse_rfc3339_test('2017-07-25T04:44:21z', 2017, 7, 25, 4, 44, 21)
    self._parse_rfc3339_test('2017-07-25T04:44:21+03:00', 2017, 7, 25, 1, 44, 21)
    self._parse_rfc3339_test('2017-07-25T04:44:21-03:00', 2017, 7, 25, 7, 44, 21)
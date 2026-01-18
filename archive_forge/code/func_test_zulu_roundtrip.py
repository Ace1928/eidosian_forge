import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_zulu_roundtrip(self):
    time_str = '2012-02-14T20:53:07Z'
    zulu = timeutils.parse_isotime(time_str)
    self.assertEqual(zulu.tzinfo, iso8601.iso8601.UTC)
    self.assertEqual(timeutils.isotime(zulu), time_str)
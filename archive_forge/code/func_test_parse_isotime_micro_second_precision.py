import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_parse_isotime_micro_second_precision(self):
    expect = timeutils.parse_isotime(self.skynet_self_aware_time_ms_str)
    skynet_self_aware_time_ms_utc = self.skynet_self_aware_ms_time.replace(tzinfo=iso8601.iso8601.UTC)
    self.assertEqual(skynet_self_aware_time_ms_utc, expect)
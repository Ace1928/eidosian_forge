import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_parse_strtime(self):
    perfect_time_format = self.skynet_self_aware_time_perfect_str
    expect = timeutils.parse_strtime(perfect_time_format)
    self.assertEqual(self.skynet_self_aware_time_perfect, expect)
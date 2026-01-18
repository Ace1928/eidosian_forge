import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_normalize_zulu_aware_to_naive(self):
    dt = datetime.datetime(2011, 2, 14, 20, 53, 7)
    time_str = '2011-02-14T19:53:07Z'
    aware = timeutils.parse_isotime(time_str)
    naive = timeutils.normalize_time(aware)
    self.assertTrue(naive < dt)
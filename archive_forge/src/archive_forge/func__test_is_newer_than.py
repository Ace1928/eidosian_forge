import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
@mock.patch('datetime.datetime', wraps=datetime.datetime)
def _test_is_newer_than(self, fn, datetime_mock):
    datetime_mock.now.return_value = self.skynet_self_aware_time
    expect_true = timeutils.is_newer_than(fn(self.one_minute_after), 59)
    self.assertTrue(expect_true)
    expect_false = timeutils.is_newer_than(fn(self.one_minute_after), 60)
    self.assertFalse(expect_false)
    expect_false = timeutils.is_newer_than(fn(self.one_minute_after), 61)
    self.assertFalse(expect_false)
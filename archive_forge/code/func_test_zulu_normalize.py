import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_zulu_normalize(self):
    time_str = '2012-02-14T20:53:07Z'
    zulu = timeutils.parse_isotime(time_str)
    normed = timeutils.normalize_time(zulu)
    self._instaneous(normed, 2012, 2, 14, 20, 53, 7, 0)
from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
@mock.patch('datetime.datetime', get_fake_datetime(datetime.datetime(2015, 12, 16, 13, 54, 26)))
@mock.patch('time.time', new=mock.Mock(return_value=1450274066.0))
@mock.patch('dateutil.tz.tzlocal', new=mock.Mock(return_value=tz.tzutc()))
def test_rfc5424_isotime_format_no_microseconds(self):
    self.config(logging_default_format_string='%(isotime)s %(message)s')
    message = 'test'
    expected = '2015-12-16T13:54:26.000000+00:00 %s\n' % message
    self.log.info(message)
    self.assertEqual(expected, self.stream.getvalue())
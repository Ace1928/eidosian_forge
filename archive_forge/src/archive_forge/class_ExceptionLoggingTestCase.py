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
class ExceptionLoggingTestCase(LogTestBase):
    """Test that Exceptions are logged."""

    def test_excepthook_logs_exception(self):
        product_name = 'somename'
        exc_log = log.getLogger(product_name)
        self._add_handler_with_cleanup(exc_log)
        excepthook = log._create_logging_excepthook(product_name)
        try:
            raise Exception('Some error happened')
        except Exception:
            excepthook(*sys.exc_info())
        expected_string = 'CRITICAL somename [-] Unhandled error: Exception: Some error happened'
        self.assertIn(expected_string, self.stream.getvalue(), message='Exception is not logged')

    def test_excepthook_installed(self):
        log.setup(self.CONF, 'test_excepthook_installed')
        self.assertTrue(sys.excepthook != sys.__excepthook__)

    @mock.patch('datetime.datetime', get_fake_datetime(datetime.datetime(2015, 12, 16, 13, 54, 26, 517893)))
    @mock.patch('dateutil.tz.tzlocal', new=mock.Mock(return_value=tz.tzutc()))
    def test_rfc5424_isotime_format(self):
        self.config(logging_default_format_string='%(isotime)s %(message)s', logging_exception_prefix='%(isotime)s ')
        product_name = 'somename'
        exc_log = log.getLogger(product_name)
        self._add_handler_with_cleanup(exc_log)
        excepthook = log._create_logging_excepthook(product_name)
        message = 'Some error happened'
        try:
            raise Exception(message)
        except Exception:
            excepthook(*sys.exc_info())
        expected_string = '2015-12-16T13:54:26.517893+00:00 Exception: %s' % message
        self.assertIn(expected_string, self.stream.getvalue())
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
@testtools.skipUnless(syslog, 'syslog is not available')
class OSSysLogHandlerTestCase(BaseTestCase):

    def test_handler(self):
        handler = handlers.OSSysLogHandler()
        syslog.syslog = mock.Mock()
        handler.emit(logging.LogRecord('foo', logging.INFO, 'path', 123, 'hey!', None, None))
        self.assertTrue(syslog.syslog.called)

    def test_syslog_binary_name(self):
        syslog.openlog = mock.Mock()
        handlers.OSSysLogHandler()
        syslog.openlog.assert_called_with(handlers._get_binary_name(), 0, syslog.LOG_USER)

    def test_find_facility(self):
        self.assertEqual(syslog.LOG_USER, log._find_facility('user'))
        self.assertEqual(syslog.LOG_LPR, log._find_facility('LPR'))
        self.assertEqual(syslog.LOG_LOCAL3, log._find_facility('log_local3'))
        self.assertEqual(syslog.LOG_UUCP, log._find_facility('LOG_UUCP'))
        self.assertRaises(TypeError, log._find_facility, 'fougere')

    def test_syslog(self):
        msg_unicode = u'Benoît Knecht & François Deppierraz login failure'
        handler = handlers.OSSysLogHandler()
        syslog.syslog = mock.Mock()
        handler.emit(logging.LogRecord('name', logging.INFO, 'path', 123, msg_unicode, None, None))
        syslog.syslog.assert_called_once_with(syslog.LOG_INFO, msg_unicode)
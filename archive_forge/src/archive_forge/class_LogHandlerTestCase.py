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
class LogHandlerTestCase(BaseTestCase):

    def test_log_path_logdir(self):
        path = os.path.join('some', 'path')
        binary = 'foo-bar'
        expected = os.path.join(path, '%s.log' % binary)
        self.config(log_dir=path, log_file=None)
        self.assertEqual(log._get_log_file_path(self.config_fixture.conf, binary=binary), expected)

    def test_log_path_logfile(self):
        path = os.path.join('some', 'path')
        binary = 'foo-bar'
        expected = os.path.join(path, '%s.log' % binary)
        self.config(log_file=expected)
        self.assertEqual(log._get_log_file_path(self.config_fixture.conf, binary=binary), expected)

    def test_log_path_none(self):
        prefix = 'foo-bar'
        self.config(log_dir=None, log_file=None)
        self.assertIsNone(log._get_log_file_path(self.config_fixture.conf, binary=prefix))

    def test_log_path_logfile_overrides_logdir(self):
        path = os.path.join(os.sep, 'some', 'path')
        prefix = 'foo-bar'
        expected = os.path.join(path, '%s.log' % prefix)
        self.config(log_dir=os.path.join('some', 'other', 'path'), log_file=expected)
        self.assertEqual(log._get_log_file_path(self.config_fixture.conf, binary=prefix), expected)

    def test_iter_loggers(self):
        mylog = logging.getLogger('abc.cde')
        loggers = list(log._iter_loggers())
        self.assertIn(logging.getLogger(), loggers)
        self.assertIn(mylog, loggers)
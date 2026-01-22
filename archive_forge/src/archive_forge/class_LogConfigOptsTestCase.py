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
class LogConfigOptsTestCase(BaseTestCase):

    def setUp(self):
        super(LogConfigOptsTestCase, self).setUp()

    def test_print_help(self):
        f = io.StringIO()
        self.CONF([])
        self.CONF.print_help(file=f)
        for option in ['debug', 'log-config', 'watch-log-file']:
            self.assertIn(option, f.getvalue())

    def test_debug(self):
        self.CONF(['--debug'])
        self.assertEqual(True, self.CONF.debug)

    def test_logging_opts(self):
        self.CONF([])
        self.assertIsNone(self.CONF.log_config_append)
        self.assertIsNone(self.CONF.log_file)
        self.assertIsNone(self.CONF.log_dir)
        self.assertEqual(_options._DEFAULT_LOG_DATE_FORMAT, self.CONF.log_date_format)
        self.assertEqual(False, self.CONF.use_syslog)
        self.assertEqual(False, self.CONF.use_json)

    def test_log_file(self):
        log_file = '/some/path/foo-bar.log'
        self.CONF(['--log-file', log_file])
        self.assertEqual(log_file, self.CONF.log_file)

    def test_log_dir_handlers(self):
        log_dir = tempfile.mkdtemp()
        self.CONF(['--log-dir', log_dir])
        self.CONF.set_default('use_stderr', False)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        logger = log._loggers[None].logger
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.handlers.WatchedFileHandler)

    def test_log_publish_errors_handlers(self):
        fake_handler = mock.MagicMock()
        with mock.patch('oslo_utils.importutils.import_object', return_value=fake_handler) as mock_import:
            log_dir = tempfile.mkdtemp()
            self.CONF(['--log-dir', log_dir])
            self.CONF.set_default('use_stderr', False)
            self.CONF.set_default('publish_errors', True)
            log._setup_logging_from_conf(self.CONF, 'test', 'test')
            logger = log._loggers[None].logger
            self.assertEqual(2, len(logger.handlers))
            self.assertIsInstance(logger.handlers[0], logging.handlers.WatchedFileHandler)
            self.assertEqual(fake_handler, logger.handlers[1])
            mock_import.assert_called_once_with('oslo_messaging.notify.log_handler.PublishErrorsHandler', logging.ERROR)

    def test_logfile_deprecated(self):
        logfile = '/some/other/path/foo-bar.log'
        self.CONF(['--logfile', logfile])
        self.assertEqual(logfile, self.CONF.log_file)

    def test_log_dir(self):
        log_dir = '/some/path/'
        self.CONF(['--log-dir', log_dir])
        self.assertEqual(log_dir, self.CONF.log_dir)

    def test_logdir_deprecated(self):
        logdir = '/some/other/path/'
        self.CONF(['--logdir', logdir])
        self.assertEqual(logdir, self.CONF.log_dir)

    def test_default_formatter(self):
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        logger = log._loggers[None].logger
        for handler in logger.handlers:
            formatter = handler.formatter
            self.assertIsInstance(formatter, formatters.ContextFormatter)

    def test_json_formatter(self):
        self.CONF(['--use-json'])
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        logger = log._loggers[None].logger
        for handler in logger.handlers:
            formatter = handler.formatter
            self.assertIsInstance(formatter, formatters.JSONFormatter)

    def test_handlers_cleanup(self):
        """Test that all old handlers get removed from log_root."""
        old_handlers = [log.handlers.ColorHandler(), log.handlers.ColorHandler()]
        log._loggers[None].logger.handlers = list(old_handlers)
        log._setup_logging_from_conf(self.CONF, 'test', 'test')
        handlers = log._loggers[None].logger.handlers
        self.assertEqual(1, len(handlers))
        self.assertNotIn(handlers[0], old_handlers)

    def test_list_opts(self):
        all_options = _options.list_opts()
        group, options = all_options[0]
        self.assertIsNone(group)
        self.assertEqual(_options.common_cli_opts + _options.logging_cli_opts + _options.generic_log_opts + _options.log_opts + _options.versionutils.deprecated_opts, options)
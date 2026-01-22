import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class LogTest(testctx.TestContextTestCase):

    def setUp(self):
        super(LogTest, self).setUp()

    def test_priv_loglevel(self):
        logger = self.useFixture(fixtures.FakeLogger(level=logging.INFO))
        logme(logging.DEBUG, 'test@DEBUG')
        logme(logging.WARN, 'test@WARN')
        time.sleep(0.1)
        self.assertNotIn('test@DEBUG', logger.output)
        self.assertIn('test@WARN', logger.output)

    def test_record_data(self):
        logs = []
        self.useFixture(fixtures.FakeLogger(level=logging.INFO, format='dummy', formatter=functools.partial(LogRecorder, logs)))
        logme(logging.WARN, 'test with exc', exc_info=True)
        time.sleep(0.1)
        self.assertEqual(1, len(logs))
        record = logs[0]
        self.assertIn('test with exc', record.getMessage())
        self.assertIsNone(record.exc_info)
        self.assertIn('TestException: with arg', record.exc_text)
        self.assertEqual('PrivContext(cfg_section=privsep)', record.processName)
        self.assertIn('test_daemon.py', record.exc_text)
        self.assertEqual(logging.WARN, record.levelno)
        self.assertEqual('logme', record.funcName)

    def test_format_record(self):
        logs = []
        self.useFixture(fixtures.FakeLogger(level=logging.INFO, format='dummy', formatter=functools.partial(LogRecorder, logs)))
        logme(logging.WARN, 'test with exc', exc_info=True)
        time.sleep(0.1)
        self.assertEqual(1, len(logs))
        record = logs[0]
        fake_config = mock.Mock(logging_default_format_string='NOCTXT: %(message)s')
        formatter = formatters.ContextFormatter(config=fake_config)
        formatter.format(record)
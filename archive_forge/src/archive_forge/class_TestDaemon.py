import logging
import os
import time
import unittest
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_privsep import comm
from oslo_privsep import priv_context
class TestDaemon(base.BaseTestCase):

    def setUp(self):
        super(TestDaemon, self).setUp()
        venv_path = os.environ['VIRTUAL_ENV']
        self.cfg_fixture = self.useFixture(config_fixture.Config())
        self.cfg_fixture.config(group='privsep', helper_command='sudo -E %s/bin/privsep-helper' % venv_path)
        priv_context.init()

    def test_concurrency(self):
        for i in range(1000):
            sleep()
        self.assertEqual(1, one())

    def test_entrypoint_with_timeout(self):
        thread_pool_size = self.cfg_fixture.conf.privsep.thread_pool_size
        for _ in range(thread_pool_size + 1):
            self.assertRaises(comm.PrivsepTimeout, sleep_with_timeout)

    def test_entrypoint_with_timeout_pass(self):
        thread_pool_size = self.cfg_fixture.conf.privsep.thread_pool_size
        for _ in range(thread_pool_size + 1):
            res = sleep_with_timeout(0.01)
            self.assertEqual(42, res)

    def test_context_with_timeout(self):
        thread_pool_size = self.cfg_fixture.conf.privsep.thread_pool_size
        for _ in range(thread_pool_size + 1):
            self.assertRaises(comm.PrivsepTimeout, sleep_with_t_context)

    def test_context_with_timeout_pass(self):
        thread_pool_size = self.cfg_fixture.conf.privsep.thread_pool_size
        for _ in range(thread_pool_size + 1):
            res = sleep_with_t_context(0.01)
            self.assertEqual(42, res)

    def test_logging(self):
        logs()
        self.assertIn('foo', self.log_fixture.logger.output)
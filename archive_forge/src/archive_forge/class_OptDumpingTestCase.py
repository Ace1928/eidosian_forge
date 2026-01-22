import argparse
import errno
import functools
import io
import logging
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import fixtures
from oslotest import base
import testscenarios
from oslo_config import cfg
from oslo_config import types
class OptDumpingTestCase(BaseTestCase):

    class FakeLogger:

        def __init__(self, test_case, expected_lvl):
            self.test_case = test_case
            self.expected_lvl = expected_lvl
            self.logged = []

        def log(self, lvl, fmt, *args):
            self.test_case.assertEqual(lvl, self.expected_lvl)
            self.logged.append(fmt % args)

    def setUp(self):
        super(OptDumpingTestCase, self).setUp()
        self._args = ['--foo', 'this', '--blaa-bar', 'that', '--blaa-key', 'admin', '--passwd', 'hush']

    def _do_test_log_opt_values(self, args):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('passwd', secret=True))
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('bar'), 'blaa')
        self.conf.register_cli_opt(cfg.StrOpt('key', secret=True), 'blaa')
        self.conf(args)
        logger = self.FakeLogger(self, 666)
        self.conf.log_opt_values(logger, 666)
        self.assertEqual(['*' * 80, 'Configuration options gathered from:', "command line args: ['--foo', 'this', '--blaa-bar', 'that', '--blaa-key', 'admin', '--passwd', 'hush']", 'config files: []', '=' * 80, 'config_dir                     = []', 'config_file                    = []', 'config_source                  = []', 'foo                            = this', 'passwd                         = ****', 'blaa.bar                       = that', 'blaa.key                       = ****', '*' * 80], logger.logged)

    def test_log_opt_values(self):
        self._do_test_log_opt_values(self._args)

    def test_log_opt_values_from_sys_argv(self):
        self.useFixture(fixtures.MonkeyPatch('sys.argv', ['foo'] + self._args))
        self._do_test_log_opt_values(None)

    def test_log_opt_values_empty_config(self):
        empty_conf = cfg.ConfigOpts()
        logger = self.FakeLogger(self, 666)
        empty_conf.log_opt_values(logger, 666)
        self.assertEqual(['*' * 80, 'Configuration options gathered from:', 'command line args: None', 'config files: []', '=' * 80, 'config_source                  = []', '*' * 80], logger.logged)
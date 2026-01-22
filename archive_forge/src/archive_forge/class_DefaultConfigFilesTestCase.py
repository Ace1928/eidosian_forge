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
class DefaultConfigFilesTestCase(BaseTestCase):

    def test_use_default(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo-', '[DEFAULT]\nfoo = bar\n')])
        self.conf.register_cli_opt(cfg.StrOpt('config-file-foo'))
        self.conf(args=['--config-file-foo', 'foo.conf'], default_config_files=[paths[0]])
        self.assertEqual([paths[0]], self.conf.config_file)
        self.assertEqual('bar', self.conf.foo)

    def test_do_not_use_default_multi_arg(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo-', '[DEFAULT]\nfoo = bar\n')])
        self.conf(args=['--config-file', paths[0]], default_config_files=['bar.conf'])
        self.assertEqual([paths[0]], self.conf.config_file)
        self.assertEqual('bar', self.conf.foo)

    def test_do_not_use_default_single_arg(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('foo-', '[DEFAULT]\nfoo = bar\n')])
        self.conf(args=['--config-file=' + paths[0]], default_config_files=['bar.conf'])
        self.assertEqual([paths[0]], self.conf.config_file)
        self.assertEqual('bar', self.conf.foo)

    def test_no_default_config_file(self):
        self.conf(args=[])
        self.assertEqual([], self.conf.config_file)

    def test_find_default_config_file(self):
        paths = self.create_tempfiles([('def', '[DEFAULT]')])
        self.useFixture(fixtures.MonkeyPatch('oslo_config.cfg.find_config_files', lambda project, prog: paths))
        self.conf(args=[], default_config_files=None)
        self.assertEqual(paths, self.conf.config_file)

    def test_default_config_file(self):
        paths = self.create_tempfiles([('def', '[DEFAULT]')])
        self.conf(args=[], default_config_files=paths)
        self.assertEqual(paths, self.conf.config_file)

    def test_default_config_file_with_value(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('def', '[DEFAULT]\nfoo = bar\n')])
        self.conf(args=[], default_config_files=paths)
        self.assertEqual(paths, self.conf.config_file)
        self.assertEqual('bar', self.conf.foo)

    def test_default_config_file_priority(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('def', '[DEFAULT]\nfoo = bar\n')])
        self.conf(args=['--foo=blaa'], default_config_files=paths)
        self.assertEqual(paths, self.conf.config_file)
        self.assertEqual('blaa', self.conf.foo)
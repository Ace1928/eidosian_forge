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
class ConfigDirTestCase(BaseTestCase):

    def test_config_dir(self):
        snafu_group = cfg.OptGroup('snafu')
        self.conf.register_group(snafu_group)
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('bell'), group=snafu_group)
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n[snafu]\nbell = whistle-00\n'), (os.path.join(dir, '02-test'), '[snafu]\nbell = whistle-02\n[DEFAULT]\nfoo = bar-02\n'), (os.path.join(dir, '01-test'), '[DEFAULT]\nfoo = bar-01\n')])
        self.conf(['--foo', 'bar', '--config-dir', os.path.dirname(paths[0])])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar-02', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'snafu'))
        self.assertTrue(hasattr(self.conf.snafu, 'bell'))
        self.assertEqual('whistle-02', self.conf.snafu.bell)

    def test_config_dir_multistr(self):
        self.conf.register_cli_opt(cfg.MultiStrOpt('foo'))
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n'), (os.path.join(dir, '02-test'), '[DEFAULT]\nfoo = bar-02\n'), (os.path.join(dir, '01-test'), '[DEFAULT]\nfoo = bar-01\n')])
        self.conf(['--foo', 'bar', '--config-dir', os.path.dirname(paths[0])])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar', 'bar-00', 'bar-01', 'bar-02'], self.conf.foo)

    def test_config_dir_file_precedence(self):
        snafu_group = cfg.OptGroup('snafu')
        self.conf.register_group(snafu_group)
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('bell'), group=snafu_group)
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n'), ('01-test', '[snafu]\nbell = whistle-01\n[DEFAULT]\nfoo = bar-01\n'), ('03-test', '[snafu]\nbell = whistle-03\n[DEFAULT]\nfoo = bar-03\n'), (os.path.join(dir, '02-test'), '[DEFAULT]\nfoo = bar-02\n')])
        self.conf(['--foo', 'bar', '--config-file', paths[1], '--config-dir', os.path.dirname(paths[0]), '--config-file', paths[2]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar-03', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'snafu'))
        self.assertTrue(hasattr(self.conf.snafu, 'bell'))
        self.assertEqual('whistle-03', self.conf.snafu.bell)

    def test_config_dir_default_file_precedence(self):
        snafu_group = cfg.OptGroup('snafu')
        self.conf.register_group(snafu_group)
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        self.conf.register_cli_opt(cfg.StrOpt('bell'), group=snafu_group)
        dir = tempfile.mkdtemp()
        self.tempdirs.append(dir)
        paths = self.create_tempfiles([(os.path.join(dir, '00-test'), '[DEFAULT]\nfoo = bar-00\n[snafu]\nbell = whistle-11\n'), ('01-test', '[snafu]\nbell = whistle-01\n[DEFAULT]\nfoo = bar-01\n'), ('03-test', '[snafu]\nbell = whistle-03\n[DEFAULT]\nfoo = bar-03\n'), (os.path.join(dir, '02-test'), '[DEFAULT]\nfoo = bar-02\n')])
        self.conf(['--foo', 'bar', '--config-dir', os.path.dirname(paths[0])], default_config_files=[paths[1], paths[2]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar-02', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'snafu'))
        self.assertTrue(hasattr(self.conf.snafu, 'bell'))
        self.assertEqual('whistle-11', self.conf.snafu.bell)

    def test_config_dir_doesnt_exist(self):
        tmpdir = tempfile.mkdtemp()
        os.rmdir(tmpdir)
        self.assertRaises(cfg.ConfigDirNotFoundError, self.conf, ['--config-dir', tmpdir])
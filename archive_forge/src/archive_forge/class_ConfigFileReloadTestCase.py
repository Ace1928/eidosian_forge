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
class ConfigFileReloadTestCase(BaseTestCase):

    def test_conf_files_reload(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = baar\n'), ('2', '[DEFAULT]\nfoo = baaar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baar', self.conf.foo)
        shutil.copy(paths[1], paths[0])
        self.conf.reload_config_files()
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baaar', self.conf.foo)

    def test_conf_files_reload_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo1'))
        self.conf.register_cli_opt(cfg.StrOpt('foo2'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo1 = default1\n'), ('2', '[DEFAULT]\nfoo2 = default2\n')])
        paths_change = self.create_tempfiles([('1', '[DEFAULT]\nfoo1 = change_default1\n'), ('2', '[DEFAULT]\nfoo2 = change_default2\n')])
        self.conf(args=[], default_config_files=paths)
        self.assertTrue(hasattr(self.conf, 'foo1'))
        self.assertEqual('default1', self.conf.foo1)
        self.assertTrue(hasattr(self.conf, 'foo2'))
        self.assertEqual('default2', self.conf.foo2)
        shutil.copy(paths_change[0], paths[0])
        shutil.copy(paths_change[1], paths[1])
        self.conf.reload_config_files()
        self.assertTrue(hasattr(self.conf, 'foo1'))
        self.assertEqual('change_default1', self.conf.foo1)
        self.assertTrue(hasattr(self.conf, 'foo2'))
        self.assertEqual('change_default2', self.conf.foo2)

    def test_conf_files_reload_file_not_found(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = baar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baar', self.conf.foo)
        os.remove(paths[0])
        self.conf.reload_config_files()
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baar', self.conf.foo)

    def test_conf_files_reload_error(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True))
        self.conf.register_cli_opt(cfg.StrOpt('foo1', required=True))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = test1\nfoo1 = test11\n'), ('2', '[DEFAULT]\nfoo2 = test2\nfoo3 = test22\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('test1', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'foo1'))
        self.assertEqual('test11', self.conf.foo1)
        shutil.copy(paths[1], paths[0])
        self.conf.reload_config_files()
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('test1', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'foo1'))
        self.assertEqual('test11', self.conf.foo1)
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
class OptGroupsTestCase(BaseTestCase):

    def test_arg_group(self):
        blaa_group = cfg.OptGroup('blaa', 'blaa options')
        self.conf.register_group(blaa_group)
        self.conf.register_cli_opt(cfg.StrOpt('foo'), group=blaa_group)
        self.conf(['--blaa-foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)
        self.assertEqual('blaa', str(blaa_group))

    def test_autocreate_group_by_name(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'), group='blaa')
        self.conf(['--blaa-foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_autocreate_group_by_group(self):
        group = cfg.OptGroup(name='blaa', title='Blaa options')
        self.conf.register_cli_opt(cfg.StrOpt('foo'), group=group)
        self.conf(['--blaa-foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_autocreate_title(self):
        blaa_group = cfg.OptGroup('blaa')
        self.assertEqual(blaa_group.title, 'blaa options')

    def test_arg_group_by_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('foo'), group='blaa')
        self.conf(['--blaa-foo', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_with_default(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='bar'), group='blaa')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_with_conf_and_group_opts(self):
        self.conf.register_cli_opt(cfg.StrOpt('conf'), group='blaa')
        self.conf.register_cli_opt(cfg.StrOpt('group'), group='blaa')
        self.conf(['--blaa-conf', 'foo', '--blaa-group', 'bar'])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'conf'))
        self.assertEqual('foo', self.conf.blaa.conf)
        self.assertTrue(hasattr(self.conf.blaa, 'group'))
        self.assertEqual('bar', self.conf.blaa.group)

    def test_arg_group_in_config_file(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo'), group='blaa')
        paths = self.create_tempfiles([('test', '[blaa]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_with_deprecated_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_name='oldfoo'), group='blaa')
        paths = self.create_tempfiles([('test', '[blaa]\noldfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_with_deprecated_group(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_group='DEFAULT'), group='blaa')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_with_deprecated_group_and_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_group='DEFAULT', deprecated_name='oldfoo'), group='blaa')
        paths = self.create_tempfiles([('test', '[DEFAULT]\noldfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_override_deprecated_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_name='oldfoo'), group='blaa')
        paths = self.create_tempfiles([('test', '[blaa]\nfoo = bar\noldfoo = blabla\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_override_deprecated_group(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_group='DEFAULT'), group='blaa')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = blabla\n[blaa]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_override_deprecated_group_and_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', deprecated_group='DEFAULT', deprecated_name='oldfoo'), group='blaa')
        paths = self.create_tempfiles([('test', '[DEFAULT]\noldfoo = blabla\n[blaa]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_with_capital_name(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo'), group='blaa')
        paths = self.create_tempfiles([('test', '[BLAA]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertFalse(hasattr(self.conf, 'BLAA'))
        self.assertTrue(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf.blaa, 'foo'))
        self.assertEqual('bar', self.conf.blaa.foo)

    def test_arg_group_in_config_file_with_capital_name_on_legacy_code(self):
        self.conf.register_group(cfg.OptGroup('BLAA'))
        self.conf.register_opt(cfg.StrOpt('foo'), group='BLAA')
        paths = self.create_tempfiles([('test', '[BLAA]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertFalse(hasattr(self.conf, 'blaa'))
        self.assertTrue(hasattr(self.conf, 'BLAA'))
        self.assertTrue(hasattr(self.conf.BLAA, 'foo'))
        self.assertEqual('bar', self.conf.BLAA.foo)
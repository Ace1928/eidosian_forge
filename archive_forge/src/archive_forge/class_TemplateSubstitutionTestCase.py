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
class TemplateSubstitutionTestCase(BaseTestCase):

    def _prep_test_str_sub(self, foo_default=None, bar_default=None):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default=foo_default))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default=bar_default))

    def _assert_str_sub(self):
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('blaa', self.conf.bar)

    def test_str_sub_default_from_default(self):
        self._prep_test_str_sub(foo_default='blaa', bar_default='$foo')
        self.conf([])
        self._assert_str_sub()

    def test_str_sub_default_from_default_recurse(self):
        self.conf.register_cli_opt(cfg.StrOpt('blaa', default='blaa'))
        self._prep_test_str_sub(foo_default='$blaa', bar_default='$foo')
        self.conf([])
        self._assert_str_sub()

    def test_str_sub_default_from_arg(self):
        self._prep_test_str_sub(bar_default='$foo')
        self.conf(['--foo', 'blaa'])
        self._assert_str_sub()

    def test_str_sub_default_from_config_file(self):
        self._prep_test_str_sub(bar_default='$foo')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = blaa\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_str_sub()

    def test_str_sub_arg_from_default(self):
        self._prep_test_str_sub(foo_default='blaa')
        self.conf(['--bar', '$foo'])
        self._assert_str_sub()

    def test_str_sub_arg_from_arg(self):
        self._prep_test_str_sub()
        self.conf(['--foo', 'blaa', '--bar', '$foo'])
        self._assert_str_sub()

    def test_str_sub_arg_from_config_file(self):
        self._prep_test_str_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = blaa\n')])
        self.conf(['--config-file', paths[0], '--bar=$foo'])
        self._assert_str_sub()

    def test_str_sub_config_file_from_default(self):
        self._prep_test_str_sub(foo_default='blaa')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_str_sub()

    def test_str_sub_config_file_from_arg(self):
        self._prep_test_str_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\n')])
        self.conf(['--config-file', paths[0], '--foo=blaa'])
        self._assert_str_sub()

    def test_str_sub_config_file_from_config_file(self):
        self._prep_test_str_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\nfoo = blaa\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_str_sub()

    def test_str_sub_with_dollar_escape_char(self):
        self._prep_test_str_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar=foo-somethin$$k2\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('foo-somethin$k2', self.conf.bar)

    def test_str_sub_with_backslash_escape_char(self):
        self._prep_test_str_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar=foo-somethin\\$k2\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('foo-somethin$k2', self.conf.bar)

    def test_str_sub_group_from_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='blaa'))
        self.conf.register_group(cfg.OptGroup('ba'))
        self.conf.register_cli_opt(cfg.StrOpt('r', default='$foo'), group='ba')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'ba'))
        self.assertTrue(hasattr(self.conf.ba, 'r'))
        self.assertEqual('blaa', self.conf.ba.r)

    def test_str_sub_set_default(self):
        self._prep_test_str_sub()
        self.conf.set_default('bar', '$foo')
        self.conf.set_default('foo', 'blaa')
        self.conf([])
        self._assert_str_sub()

    def test_str_sub_set_override(self):
        self._prep_test_str_sub()
        self.conf.set_override('bar', '$foo')
        self.conf.set_override('foo', 'blaa')
        self.conf([])
        self._assert_str_sub()

    def _prep_test_str_int_sub(self, foo_default=None, bar_default=None):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default=foo_default))
        self.conf.register_cli_opt(cfg.IntOpt('bar', default=bar_default))

    def _assert_int_sub(self):
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual(123, self.conf.bar)

    def test_sub_default_from_default(self):
        self._prep_test_str_int_sub(foo_default='123', bar_default='$foo')
        self.conf([])
        self._assert_int_sub()

    def test_sub_default_from_default_recurse(self):
        self.conf.register_cli_opt(cfg.StrOpt('blaa', default='123'))
        self._prep_test_str_int_sub(foo_default='$blaa', bar_default='$foo')
        self.conf([])
        self._assert_int_sub()

    def test_sub_default_from_arg(self):
        self._prep_test_str_int_sub(bar_default='$foo')
        self.conf(['--foo', '123'])
        self._assert_int_sub()

    def test_sub_default_from_config_file(self):
        self._prep_test_str_int_sub(bar_default='$foo')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 123\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_int_sub()

    def test_sub_arg_from_default(self):
        self._prep_test_str_int_sub(foo_default='123')
        self.conf(['--bar', '$foo'])
        self._assert_int_sub()

    def test_sub_arg_from_arg(self):
        self._prep_test_str_int_sub()
        self.conf(['--foo', '123', '--bar', '$foo'])
        self._assert_int_sub()

    def test_sub_arg_from_config_file(self):
        self._prep_test_str_int_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 123\n')])
        self.conf(['--config-file', paths[0], '--bar=$foo'])
        self._assert_int_sub()

    def test_sub_config_file_from_default(self):
        self._prep_test_str_int_sub(foo_default='123')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_int_sub()

    def test_sub_config_file_from_arg(self):
        self._prep_test_str_int_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\n')])
        self.conf(['--config-file', paths[0], '--foo=123'])
        self._assert_int_sub()

    def test_sub_config_file_from_config_file(self):
        self._prep_test_str_int_sub()
        paths = self.create_tempfiles([('test', '[DEFAULT]\nbar = $foo\nfoo = 123\n')])
        self.conf(['--config-file', paths[0]])
        self._assert_int_sub()

    def test_sub_group_from_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='123'))
        self.conf.register_group(cfg.OptGroup('ba'))
        self.conf.register_cli_opt(cfg.IntOpt('r', default='$foo'), group='ba')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'ba'))
        self.assertTrue(hasattr(self.conf.ba, 'r'))
        self.assertEqual('123', self.conf.foo)
        self.assertEqual(123, self.conf.ba.r)

    def test_sub_group_from_default_deprecated(self):
        self.conf.register_group(cfg.OptGroup('ba'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='123', deprecated_group='DEFAULT'), group='ba')
        self.conf.register_cli_opt(cfg.IntOpt('r', default='$foo'), group='ba')
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'ba'))
        self.assertTrue(hasattr(self.conf.ba, 'foo'))
        self.assertEqual('123', self.conf.ba.foo)
        self.assertTrue(hasattr(self.conf.ba, 'r'))
        self.assertEqual(123, self.conf.ba.r)

    def test_sub_group_from_args_deprecated(self):
        self.conf.register_group(cfg.OptGroup('ba'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='123', deprecated_group='DEFAULT'), group='ba')
        self.conf.register_cli_opt(cfg.IntOpt('r', default='$foo'), group='ba')
        self.conf(['--ba-foo=4242'])
        self.assertTrue(hasattr(self.conf, 'ba'))
        self.assertTrue(hasattr(self.conf.ba, 'foo'))
        self.assertTrue(hasattr(self.conf.ba, 'r'))
        self.assertEqual('4242', self.conf.ba.foo)
        self.assertEqual(4242, self.conf.ba.r)

    def test_sub_group_from_configfile_deprecated(self):
        self.conf.register_group(cfg.OptGroup('ba'))
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='123', deprecated_group='DEFAULT'), group='ba')
        self.conf.register_cli_opt(cfg.IntOpt('r', default='$foo'), group='ba')
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo=4242\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'ba'))
        self.assertTrue(hasattr(self.conf.ba, 'foo'))
        self.assertTrue(hasattr(self.conf.ba, 'r'))
        self.assertEqual('4242', self.conf.ba.foo)
        self.assertEqual(4242, self.conf.ba.r)

    def test_dict_sub_default_from_default(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='floo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='blaa'))
        self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar'}))
        self.conf([])
        self.assertEqual('blaa', self.conf.dt['floo'])

    def test_dict_sub_default_from_default_multi(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='floo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='blaa'))
        self.conf.register_cli_opt(cfg.StrOpt('goo', default='gloo'))
        self.conf.register_cli_opt(cfg.StrOpt('har', default='hlaa'))
        self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar', '$goo': 'goo', 'har': '$har', 'key1': 'str', 'key2': 12345}))
        self.conf([])
        self.assertEqual('blaa', self.conf.dt['floo'])
        self.assertEqual('goo', self.conf.dt['gloo'])
        self.assertEqual('hlaa', self.conf.dt['har'])
        self.assertEqual('str', self.conf.dt['key1'])
        self.assertEqual(12345, self.conf.dt['key2'])

    def test_dict_sub_default_from_default_recurse(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='$foo2'))
        self.conf.register_cli_opt(cfg.StrOpt('foo2', default='floo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='$bar2'))
        self.conf.register_cli_opt(cfg.StrOpt('bar2', default='blaa'))
        self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar'}))
        self.conf([])
        self.assertEqual('blaa', self.conf.dt['floo'])

    def test_dict_sub_default_from_arg(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default=None))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default=None))
        self.conf.register_cli_opt(cfg.DictOpt('dt', default={'$foo': '$bar'}))
        self.conf(['--foo', 'floo', '--bar', 'blaa'])
        self.assertTrue(hasattr(self.conf, 'dt'))
        self.assertEqual('blaa', self.conf.dt['floo'])

    def test_dict_sub_default_from_config_file(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', default='floo'))
        self.conf.register_cli_opt(cfg.StrOpt('bar', default='blaa'))
        self.conf.register_cli_opt(cfg.DictOpt('dt', default={}))
        paths = self.create_tempfiles([('test', '[DEFAULT]\ndt = $foo:$bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'dt'))
        self.assertEqual('blaa', self.conf.dt['floo'])
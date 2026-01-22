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
class PositionalTestCase(BaseTestCase):

    def _do_pos_test(self, opt_class, default, cli_args, value):
        self.conf.register_cli_opt(opt_class('foo', default=default, positional=True, required=False))
        self.conf(cli_args)
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(value, self.conf.foo)

    def test_positional_str_none_default(self):
        self._do_pos_test(cfg.StrOpt, None, [], None)

    def test_positional_str_default(self):
        self._do_pos_test(cfg.StrOpt, 'bar', [], 'bar')

    def test_positional_str_arg(self):
        self._do_pos_test(cfg.StrOpt, None, ['bar'], 'bar')

    def test_positional_int_none_default(self):
        self._do_pos_test(cfg.IntOpt, None, [], None)

    def test_positional_int_default(self):
        self._do_pos_test(cfg.IntOpt, 10, [], 10)

    def test_positional_int_arg(self):
        self._do_pos_test(cfg.IntOpt, None, ['20'], 20)

    def test_positional_float_none_default(self):
        self._do_pos_test(cfg.FloatOpt, None, [], None)

    def test_positional_float_default(self):
        self._do_pos_test(cfg.FloatOpt, 1.0, [], 1.0)

    def test_positional_float_arg(self):
        self._do_pos_test(cfg.FloatOpt, None, ['2.0'], 2.0)

    def test_positional_list_none_default(self):
        self._do_pos_test(cfg.ListOpt, None, [], None)

    def test_positional_list_empty_default(self):
        self._do_pos_test(cfg.ListOpt, [], [], [])

    def test_positional_list_default(self):
        self._do_pos_test(cfg.ListOpt, ['bar'], [], ['bar'])

    def test_positional_list_arg(self):
        self._do_pos_test(cfg.ListOpt, None, ['blaa,bar'], ['blaa', 'bar'])

    def test_positional_dict_none_default(self):
        self._do_pos_test(cfg.DictOpt, None, [], None)

    def test_positional_dict_empty_default(self):
        self._do_pos_test(cfg.DictOpt, {}, [], {})

    def test_positional_dict_default(self):
        self._do_pos_test(cfg.DictOpt, {'key1': 'bar'}, [], {'key1': 'bar'})

    def test_positional_dict_arg(self):
        self._do_pos_test(cfg.DictOpt, None, ['key1:blaa,key2:bar'], {'key1': 'blaa', 'key2': 'bar'})

    def test_positional_ip_none_default(self):
        self._do_pos_test(cfg.IPOpt, None, [], None)

    def test_positional_ip_default(self):
        self._do_pos_test(cfg.IPOpt, '127.0.0.1', [], '127.0.0.1')

    def test_positional_ip_arg(self):
        self._do_pos_test(cfg.IPOpt, None, ['127.0.0.1'], '127.0.0.1')

    def test_positional_port_none_default(self):
        self._do_pos_test(cfg.PortOpt, None, [], None)

    def test_positional_port_default(self):
        self._do_pos_test(cfg.PortOpt, 80, [], 80)

    def test_positional_port_arg(self):
        self._do_pos_test(cfg.PortOpt, None, ['443'], 443)

    def test_positional_uri_default(self):
        self._do_pos_test(cfg.URIOpt, 'http://example.com', [], 'http://example.com')

    def test_positional_uri_none_default(self):
        self._do_pos_test(cfg.URIOpt, None, [], None)

    def test_positional_uri_arg(self):
        self._do_pos_test(cfg.URIOpt, None, ['http://example.com'], 'http://example.com')

    def test_positional_multistr_none_default(self):
        self._do_pos_test(cfg.MultiStrOpt, None, [], None)

    def test_positional_multistr_empty_default(self):
        self._do_pos_test(cfg.MultiStrOpt, [], [], [])

    def test_positional_multistr_default(self):
        self._do_pos_test(cfg.MultiStrOpt, ['bar'], [], ['bar'])

    def test_positional_multistr_arg(self):
        self._do_pos_test(cfg.MultiStrOpt, None, ['blaa', 'bar'], ['blaa', 'bar'])

    def test_positional_bool(self):
        self.assertRaises(ValueError, cfg.BoolOpt, 'foo', positional=True)

    def test_required_positional_opt_defined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' foo\n', sys.stdout.getvalue())
        self.conf(['bar'])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_required_positional_opt_undefined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' foo\n', sys.stdout.getvalue())
        self.assertRaises(SystemExit, self.conf, [])

    def test_optional_positional_opt_defined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=False, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' [foo]\n', sys.stdout.getvalue())
        self.conf(['bar'])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_optional_positional_opt_undefined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=False, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' [foo]\n', sys.stdout.getvalue())
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertIsNone(self.conf.foo)

    def test_optional_positional_hyphenated_opt_defined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=False, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' [foo_bar]\n', sys.stdout.getvalue())
        self.conf(['baz'])
        self.assertTrue(hasattr(self.conf, 'foo_bar'))
        self.assertEqual('baz', self.conf.foo_bar)

    def test_optional_positional_hyphenated_opt_undefined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=False, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' [foo_bar]\n', sys.stdout.getvalue())
        self.conf([])
        self.assertTrue(hasattr(self.conf, 'foo_bar'))
        self.assertIsNone(self.conf.foo_bar)

    def test_required_positional_hyphenated_opt_defined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=True, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' foo_bar\n', sys.stdout.getvalue())
        self.conf(['baz'])
        self.assertTrue(hasattr(self.conf, 'foo_bar'))
        self.assertEqual('baz', self.conf.foo_bar)

    def test_required_positional_hyphenated_opt_undefined(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo-bar', required=True, positional=True))
        self.useFixture(fixtures.MonkeyPatch('sys.stdout', io.StringIO()))
        self.assertRaises(SystemExit, self.conf, ['--help'])
        self.assertIn(' foo_bar\n', sys.stdout.getvalue())
        self.assertRaises(SystemExit, self.conf, [])

    def test_missing_required_cli_opt(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo', required=True, positional=True))
        self.assertRaises(SystemExit, self.conf, [])

    def test_positional_opts_order(self):
        self.conf.register_cli_opts((cfg.StrOpt('command', positional=True), cfg.StrOpt('arg1', positional=True), cfg.StrOpt('arg2', positional=True)))
        self.conf(['command', 'arg1', 'arg2'])
        self.assertEqual('command', self.conf.command)
        self.assertEqual('arg1', self.conf.arg1)
        self.assertEqual('arg2', self.conf.arg2)

    def test_positional_opt_order(self):
        self.conf.register_cli_opt(cfg.StrOpt('command', positional=True))
        self.conf.register_cli_opt(cfg.StrOpt('arg1', positional=True))
        self.conf.register_cli_opt(cfg.StrOpt('arg2', positional=True))
        self.conf(['command', 'arg1', 'arg2'])
        self.assertEqual('command', self.conf.command)
        self.assertEqual('arg1', self.conf.arg1)
        self.assertEqual('arg2', self.conf.arg2)

    def test_positional_opt_unregister(self):
        command = cfg.StrOpt('command', positional=True)
        arg1 = cfg.StrOpt('arg1', positional=True)
        arg2 = cfg.StrOpt('arg2', positional=True)
        self.conf.register_cli_opt(command)
        self.conf.register_cli_opt(arg1)
        self.conf.register_cli_opt(arg2)
        self.conf(['command', 'arg1', 'arg2'])
        self.assertEqual('command', self.conf.command)
        self.assertEqual('arg1', self.conf.arg1)
        self.assertEqual('arg2', self.conf.arg2)
        self.conf.reset()
        self.conf.unregister_opt(arg1)
        self.conf.unregister_opt(arg2)
        arg0 = cfg.StrOpt('arg0', positional=True)
        self.conf.register_cli_opt(arg0)
        self.conf.register_cli_opt(arg1)
        self.conf(['command', 'arg0', 'arg1'])
        self.assertEqual('command', self.conf.command)
        self.assertEqual('arg0', self.conf.arg0)
        self.assertEqual('arg1', self.conf.arg1)
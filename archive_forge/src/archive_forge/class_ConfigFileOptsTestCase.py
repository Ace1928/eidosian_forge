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
@mock.patch('oslo_log.versionutils.report_deprecated_feature', _fake_deprecated_feature)
class ConfigFileOptsTestCase(BaseTestCase):

    def setUp(self):
        super(ConfigFileOptsTestCase, self).setUp()
        self.logger = self.useFixture(fixtures.FakeLogger(format='%(message)s', level=logging.WARNING, nuke_handlers=True))

    def _do_deprecated_test(self, opt_class, value, result, key, section='DEFAULT', dname=None, dgroup=None):
        self.conf.register_opt(opt_class('newfoo', deprecated_name=dname, deprecated_group=dgroup))
        paths = self.create_tempfiles([('test', '[' + section + ']\n' + key + ' = ' + value + '\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'newfoo'))
        self.assertEqual(result, self.conf.newfoo)

    def _do_dname_test_use(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'oldfoo', dname='oldfoo')

    def _do_dgroup_test_use(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'newfoo', section='old', dgroup='old')

    def _do_default_dgroup_test_use(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'newfoo', section='DEFAULT', dgroup='DEFAULT')

    def _do_dgroup_and_dname_test_use(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'oof', section='old', dgroup='old', dname='oof')

    def _do_dname_test_ignore(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'newfoo', dname='oldfoo')

    def _do_dgroup_test_ignore(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'newfoo', section='DEFAULT', dgroup='old')

    def _do_dgroup_and_dname_test_ignore(self, opt_class, value, result):
        self._do_deprecated_test(opt_class, value, result, 'oof', section='old', dgroup='old', dname='oof')

    def test_conf_file_str_default(self):
        self.conf.register_opt(cfg.StrOpt('foo', default='bar'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_conf_file_str_value(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_conf_file_str_value_override(self):
        self.conf.register_cli_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = baar\n'), ('2', '[DEFAULT]\nfoo = baaar\n')])
        self.conf(['--foo', 'bar', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('baaar', self.conf.foo)

    def test_conf_file_str_value_override_use_deprecated(self):
        """last option should always win, even if last uses deprecated."""
        self.conf.register_cli_opt(cfg.StrOpt('newfoo', deprecated_name='oldfoo'))
        paths = self.create_tempfiles([('0', '[DEFAULT]\nnewfoo = middle\n'), ('1', '[DEFAULT]\noldfoo = last\n')])
        self.conf(['--newfoo', 'first', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'newfoo'))
        self.assertTrue(hasattr(self.conf, 'oldfoo'))
        self.assertEqual('last', self.conf.newfoo)
        log_out = self.logger.output
        self.assertIn('is deprecated', log_out)
        self.assertIn('Use option "newfoo"', log_out)

    def test_use_deprecated_for_removal_without_reason(self):
        self.conf.register_cli_opt(cfg.StrOpt('oldfoo', deprecated_for_removal=True))
        paths = self.create_tempfiles([('0', '[DEFAULT]\noldfoo = middle\n')])
        self.conf(['--oldfoo', 'first', '--config-file', paths[0]])
        log_out = self.logger.output
        self.assertIn('deprecated for removal.', log_out)

    def test_use_deprecated_for_removal_with_reason(self):
        self.conf.register_cli_opt(cfg.StrOpt('oldfoo', deprecated_for_removal=True, deprecated_reason='a very good reason'))
        paths = self.create_tempfiles([('0', '[DEFAULT]\noldfoo = middle\n')])
        self.conf(['--oldfoo', 'first', '--config-file', paths[0]])
        log_out = self.logger.output
        self.assertIn('deprecated for removal (a very good reason).', log_out)

    def test_conf_file_str_use_dname(self):
        self._do_dname_test_use(cfg.StrOpt, 'value1', 'value1')

    def test_conf_file_str_use_dgroup(self):
        self._do_dgroup_test_use(cfg.StrOpt, 'value1', 'value1')

    def test_conf_file_str_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.StrOpt, 'value1', 'value1')

    def test_conf_file_str_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.StrOpt, 'value1', 'value1')

    def test_conf_file_str_ignore_dname(self):
        self._do_dname_test_ignore(cfg.StrOpt, 'value2', 'value2')

    def test_conf_file_str_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.StrOpt, 'value2', 'value2')

    def test_conf_file_str_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.StrOpt, 'value2', 'value2')

    def test_conf_file_str_value_with_good_choice_value(self):
        self.conf.register_opt(cfg.StrOpt('foo', choices=['bar', 'baz']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)

    def test_conf_file_bool_default_none(self):
        self.conf.register_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertIsNone(self.conf.foo)

    def test_conf_file_bool_default_false(self):
        self.conf.register_opt(cfg.BoolOpt('foo', default=False))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertFalse(self.conf.foo)

    def test_conf_file_bool_value(self):
        self.conf.register_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = true\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertTrue(self.conf.foo)

    def test_conf_file_bool_cli_value_override(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 0\n')])
        self.conf(['--foo', '--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertFalse(self.conf.foo)

    def test_conf_file_bool_cli_inverse_override(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = true\n')])
        self.conf(['--nofoo', '--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertTrue(self.conf.foo)

    def test_conf_file_bool_cli_order_override(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = false\n')])
        self.conf(['--config-file', paths[0], '--foo'])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertTrue(self.conf.foo)

    def test_conf_file_bool_file_value_override(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 0\n'), ('2', '[DEFAULT]\nfoo = yes\n')])
        self.conf(['--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertTrue(self.conf.foo)

    def test_conf_file_bool_use_dname(self):
        self._do_dname_test_use(cfg.BoolOpt, 'yes', True)

    def test_conf_file_bool_use_dgroup(self):
        self._do_dgroup_test_use(cfg.BoolOpt, 'yes', True)

    def test_conf_file_bool_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.BoolOpt, 'yes', True)

    def test_conf_file_bool_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.BoolOpt, 'yes', True)

    def test_conf_file_bool_ignore_dname(self):
        self._do_dname_test_ignore(cfg.BoolOpt, 'no', False)

    def test_conf_file_bool_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.BoolOpt, 'no', False)

    def test_conf_file_bool_ignore_group_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.BoolOpt, 'no', False)

    def test_conf_file_int_default(self):
        self.conf.register_opt(cfg.IntOpt('foo', default=666))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(666, self.conf.foo)

    def test_conf_file_int_string_default_type(self):
        self.conf.register_opt(cfg.IntOpt('foo', default='666'))
        self.conf([])
        self.assertEqual(666, self.conf.foo)

    def test_conf_file_int_value(self):
        self.conf.register_opt(cfg.IntOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 666\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(666, self.conf.foo)

    def test_conf_file_int_value_override(self):
        self.conf.register_cli_opt(cfg.IntOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 66\n'), ('2', '[DEFAULT]\nfoo = 666\n')])
        self.conf(['--foo', '6', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(666, self.conf.foo)

    def test_conf_file_int_min_max(self):
        self.conf.register_opt(cfg.IntOpt('foo', min=1, max=5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 10\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_int_min_greater_max(self):
        self.assertRaises(ValueError, cfg.IntOpt, 'foo', min=5, max=1)

    def test_conf_file_int_use_dname(self):
        self._do_dname_test_use(cfg.IntOpt, '66', 66)

    def test_conf_file_int_use_dgroup(self):
        self._do_dgroup_test_use(cfg.IntOpt, '66', 66)

    def test_conf_file_int_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.IntOpt, '66', 66)

    def test_conf_file_int_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.IntOpt, '66', 66)

    def test_conf_file_int_ignore_dname(self):
        self._do_dname_test_ignore(cfg.IntOpt, '64', 64)

    def test_conf_file_int_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.IntOpt, '64', 64)

    def test_conf_file_int_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.IntOpt, '64', 64)

    def test_conf_file_float_default(self):
        self.conf.register_opt(cfg.FloatOpt('foo', default=6.66))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(6.66, self.conf.foo)

    def test_conf_file_float_default_wrong_type(self):
        self.assertRaises(cfg.DefaultValueError, cfg.FloatOpt, 'foo', default='foobar6.66')

    def test_conf_file_float_value(self):
        self.conf.register_opt(cfg.FloatOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 6.66\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(6.66, self.conf.foo)

    def test_conf_file_float_value_override(self):
        self.conf.register_cli_opt(cfg.FloatOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 6.6\n'), ('2', '[DEFAULT]\nfoo = 6.66\n')])
        self.conf(['--foo', '6', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(6.66, self.conf.foo)

    def test_conf_file_float_use_dname(self):
        self._do_dname_test_use(cfg.FloatOpt, '66.54', 66.54)

    def test_conf_file_float_use_dgroup(self):
        self._do_dgroup_test_use(cfg.FloatOpt, '66.54', 66.54)

    def test_conf_file_float_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.FloatOpt, '66.54', 66.54)

    def test_conf_file_float_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.FloatOpt, '66.54', 66.54)

    def test_conf_file_float_ignore_dname(self):
        self._do_dname_test_ignore(cfg.FloatOpt, '64.54', 64.54)

    def test_conf_file_float_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.FloatOpt, '64.54', 64.54)

    def test_conf_file_float_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.FloatOpt, '64.54', 64.54)

    def test_conf_file_float_min_max_above_max(self):
        self.conf.register_opt(cfg.FloatOpt('foo', min=1.1, max=5.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 10.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_float_only_max_above_max(self):
        self.conf.register_opt(cfg.FloatOpt('foo', max=5.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 10.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_float_min_max_below_min(self):
        self.conf.register_opt(cfg.FloatOpt('foo', min=1.1, max=5.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 0.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_float_only_min_below_min(self):
        self.conf.register_opt(cfg.FloatOpt('foo', min=1.1))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 0.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_float_min_max_in_range(self):
        self.conf.register_opt(cfg.FloatOpt('foo', min=1.1, max=5.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 4.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(4.5, self.conf.foo)

    def test_conf_file_float_only_max_in_range(self):
        self.conf.register_opt(cfg.FloatOpt('foo', max=5.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 4.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(4.5, self.conf.foo)

    def test_conf_file_float_only_min_in_range(self):
        self.conf.register_opt(cfg.FloatOpt('foo', min=3.5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 4.5\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(4.5, self.conf.foo)

    def test_conf_file_float_min_greater_max(self):
        self.assertRaises(ValueError, cfg.FloatOpt, 'foo', min=5.5, max=1.5)

    def test_conf_file_list_default(self):
        self.conf.register_opt(cfg.ListOpt('foo', default=['bar']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar'], self.conf.foo)

    def test_conf_file_list_default_wrong_type(self):
        self.assertRaises(cfg.DefaultValueError, cfg.ListOpt, 'foo', default=25)

    def test_conf_file_list_value(self):
        self.conf.register_opt(cfg.ListOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar'], self.conf.foo)

    def test_conf_file_list_value_override(self):
        self.conf.register_cli_opt(cfg.ListOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = bar,bar\n'), ('2', '[DEFAULT]\nfoo = b,a,r\n')])
        self.conf(['--foo', 'bar', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['b', 'a', 'r'], self.conf.foo)

    def test_conf_file_list_item_type(self):
        self.conf.register_cli_opt(cfg.ListOpt('foo', item_type=types.Integer()))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = 1,2\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual([1, 2], self.conf.foo)

    def test_conf_file_list_item_wrong_type(self):
        self.assertRaises(cfg.DefaultValueError, cfg.ListOpt, 'foo', default='bar', item_type=types.Integer())

    def test_conf_file_list_bounds(self):
        self.conf.register_cli_opt(cfg.ListOpt('foo', item_type=types.Integer(), default='[1,2]', bounds=True))
        self.conf([])
        self.assertEqual([1, 2], self.conf.foo)

    def test_conf_file_list_use_dname(self):
        self._do_dname_test_use(cfg.ListOpt, 'a,b,c', ['a', 'b', 'c'])

    def test_conf_file_list_use_dgroup(self):
        self._do_dgroup_test_use(cfg.ListOpt, 'a,b,c', ['a', 'b', 'c'])

    def test_conf_file_list_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.ListOpt, 'a,b,c', ['a', 'b', 'c'])

    def test_conf_file_list_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.ListOpt, 'a,b,c', ['a', 'b', 'c'])

    def test_conf_file_list_ignore_dname(self):
        self._do_dname_test_ignore(cfg.ListOpt, 'd,e,f', ['d', 'e', 'f'])

    def test_conf_file_list_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.ListOpt, 'd,e,f', ['d', 'e', 'f'])

    def test_conf_file_list_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.ListOpt, 'd,e,f', ['d', 'e', 'f'])

    def test_conf_file_list_spaces_use_dname(self):
        self._do_dname_test_use(cfg.ListOpt, 'a, b, c', ['a', 'b', 'c'])

    def test_conf_file_list_spaces_use_dgroup(self):
        self._do_dgroup_test_use(cfg.ListOpt, 'a, b, c', ['a', 'b', 'c'])

    def test_conf_file_list_spaces_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.ListOpt, 'a, b, c', ['a', 'b', 'c'])

    def test_conf_file_list_spaces_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.ListOpt, 'a, b, c', ['a', 'b', 'c'])

    def test_conf_file_list_spaces_ignore_dname(self):
        self._do_dname_test_ignore(cfg.ListOpt, 'd, e, f', ['d', 'e', 'f'])

    def test_conf_file_list_spaces_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.ListOpt, 'd, e, f', ['d', 'e', 'f'])

    def test_conf_file_list_spaces_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.ListOpt, 'd, e, f', ['d', 'e', 'f'])

    def test_conf_file_dict_default(self):
        self.conf.register_opt(cfg.DictOpt('foo', default={'key': 'bar'}))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual({'key': 'bar'}, self.conf.foo)

    def test_conf_file_dict_value(self):
        self.conf.register_opt(cfg.DictOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = key:bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual({'key': 'bar'}, self.conf.foo)

    def test_conf_file_dict_colon_in_value(self):
        self.conf.register_opt(cfg.DictOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = key:bar:baz\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual({'key': 'bar:baz'}, self.conf.foo)

    def test_conf_file_dict_value_no_colon(self):
        self.conf.register_opt(cfg.DictOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = key:bar,baz\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')
        self.assertRaises(ValueError, getattr, self.conf, 'foo')

    def test_conf_file_dict_value_duplicate_key(self):
        self.conf.register_opt(cfg.DictOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = key:bar,key:baz\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')
        self.assertRaises(ValueError, getattr, self.conf, 'foo')

    def test_conf_file_dict_values_override_deprecated(self):
        self.conf.register_cli_opt(cfg.DictOpt('foo', deprecated_name='oldfoo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = key1:bar1\n'), ('2', '[DEFAULT]\noldfoo = key2:bar2\noldfoo = key3:bar3\n')])
        self.conf(['--foo', 'key0:bar0', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual({'key3': 'bar3'}, self.conf.foo)

    def test_conf_file_dict_deprecated(self):
        self.conf.register_opt(cfg.DictOpt('newfoo', deprecated_name='oldfoo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\noldfoo= key1:bar1\noldfoo = key2:bar2\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'newfoo'))
        self.assertEqual({'key2': 'bar2'}, self.conf.newfoo)

    def test_conf_file_dict_value_override(self):
        self.conf.register_cli_opt(cfg.DictOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = key:bar,key2:bar\n'), ('2', '[DEFAULT]\nfoo = k1:v1,k2:v2\n')])
        self.conf(['--foo', 'x:y,x2:y2', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual({'k1': 'v1', 'k2': 'v2'}, self.conf.foo)

    def test_conf_file_dict_use_dname(self):
        self._do_dname_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_use_dgroup(self):
        self._do_dgroup_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_ignore_dname(self):
        self._do_dname_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_dict_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_dict_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_dict_spaces_use_dname(self):
        self._do_dname_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_spaces_use_dgroup(self):
        self._do_dgroup_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_spaces_use_default_dgroup(self):
        self._do_default_dgroup_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_spaces_use_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_use(cfg.DictOpt, 'k1:a,k2:b,k3:c', {'k1': 'a', 'k2': 'b', 'k3': 'c'})

    def test_conf_file_dict_spaces_ignore_dname(self):
        self._do_dname_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_dict_spaces_ignore_dgroup(self):
        self._do_dgroup_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_dict_spaces_ignore_dgroup_and_dname(self):
        self._do_dgroup_and_dname_test_ignore(cfg.DictOpt, 'k1:d,k2:e,k3:f', {'k1': 'd', 'k2': 'e', 'k3': 'f'})

    def test_conf_file_port_outside_range(self):
        self.conf.register_opt(cfg.PortOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 65536\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_list(self):
        self.conf.register_opt(cfg.ListOpt('foo', item_type=types.Port()))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 22, 80\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual([22, 80], self.conf.foo)

    def test_conf_file_port_list_default(self):
        self.conf.register_opt(cfg.ListOpt('foo', item_type=types.Port(), default=[55, 77]))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 22, 80\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual([22, 80], self.conf.foo)

    def test_conf_file_port_list_only_default(self):
        self.conf.register_opt(cfg.ListOpt('foo', item_type=types.Port(), default=[55, 77]))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual([55, 77], self.conf.foo)

    def test_conf_file_port_list_outside_range(self):
        self.conf.register_opt(cfg.ListOpt('foo', item_type=types.Port()))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 1,65536\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_min_max_above_max(self):
        self.conf.register_opt(cfg.PortOpt('foo', min=1, max=5))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 10\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_only_max_above_max(self):
        self.conf.register_opt(cfg.PortOpt('foo', max=500))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 600\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_min_max_below_min(self):
        self.conf.register_opt(cfg.PortOpt('foo', min=100, max=500))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 99\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_only_min_below_min(self):
        self.conf.register_opt(cfg.PortOpt('foo', min=1025))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 1024\n')])
        self.conf(['--config-file', paths[0]])
        self.assertRaises(cfg.ConfigFileValueError, self.conf._get, 'foo')

    def test_conf_file_port_min_max_in_range(self):
        self.conf.register_opt(cfg.PortOpt('foo', min=1025, max=6000))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 2500\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(2500, self.conf.foo)

    def test_conf_file_port_only_max_in_range(self):
        self.conf.register_opt(cfg.PortOpt('foo', max=5000))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 45\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(45, self.conf.foo)

    def test_conf_file_port_only_min_in_range(self):
        self.conf.register_opt(cfg.PortOpt('foo', min=35))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = 45\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(45, self.conf.foo)

    def test_conf_file_port_min_greater_max(self):
        self.assertRaises(ValueError, cfg.PortOpt, 'foo', min=55, max=15)

    def test_conf_file_multistr_default(self):
        self.conf.register_opt(cfg.MultiStrOpt('foo', default=['bar']))
        paths = self.create_tempfiles([('test', '[DEFAULT]\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar'], self.conf.foo)

    def test_conf_file_multistr_value(self):
        self.conf.register_opt(cfg.MultiStrOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar'], self.conf.foo)

    def test_conf_file_multistr_values_append_deprecated(self):
        self.conf.register_cli_opt(cfg.MultiStrOpt('foo', deprecated_name='oldfoo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = bar1\n'), ('2', '[DEFAULT]\noldfoo = bar2\noldfoo = bar3\n')])
        self.conf(['--foo', 'bar0', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar0', 'bar1', 'bar2', 'bar3'], self.conf.foo)

    def test_conf_file_multistr_values_append(self):
        self.conf.register_cli_opt(cfg.MultiStrOpt('foo'))
        paths = self.create_tempfiles([('1', '[DEFAULT]\nfoo = bar1\n'), ('2', '[DEFAULT]\nfoo = bar2\nfoo = bar3\n')])
        self.conf(['--foo', 'bar0', '--config-file', paths[0], '--config-file', paths[1]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual(['bar0', 'bar1', 'bar2', 'bar3'], self.conf.foo)

    def test_conf_file_multistr_deprecated(self):
        self.conf.register_opt(cfg.MultiStrOpt('newfoo', deprecated_name='oldfoo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\noldfoo= bar1\noldfoo = bar2\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'newfoo'))
        self.assertEqual(['bar1', 'bar2'], self.conf.newfoo)

    def test_conf_file_multiple_opts(self):
        self.conf.register_opts([cfg.StrOpt('foo'), cfg.StrOpt('bar')])
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar\nbar = foo\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar', self.conf.foo)
        self.assertTrue(hasattr(self.conf, 'bar'))
        self.assertEqual('foo', self.conf.bar)

    def test_conf_file_raw_value(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        paths = self.create_tempfiles([('test', '[DEFAULT]\nfoo = bar-%08x\n')])
        self.conf(['--config-file', paths[0]])
        self.assertTrue(hasattr(self.conf, 'foo'))
        self.assertEqual('bar-%08x', self.conf.foo)

    def test_conf_file_sorted_group(self):
        for i in range(10):
            group = cfg.OptGroup('group%s' % i, 'options')
            self.conf.register_group(group)
            self.conf.register_cli_opt(cfg.StrOpt('opt1'), group=group)
        paths = self.create_tempfiles([('test', '[group1]\nopt1 = foo\n[group2]\nopt2 = bar\n')])
        self.conf(['--config-file', paths[0]])
        self.assertEqual('foo', self.conf.group1.opt1)
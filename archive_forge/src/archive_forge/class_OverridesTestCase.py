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
class OverridesTestCase(BaseTestCase):

    def test_default_none(self):
        self.conf.register_opt(cfg.StrOpt('foo', default='foo'))
        self.conf([])
        self.assertEqual('foo', self.conf.foo)
        self.conf.set_default('foo', None)
        self.assertIsNone(self.conf.foo)
        self.conf.clear_default('foo')
        self.assertEqual('foo', self.conf.foo)

    def test_no_default_override(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        self.conf([])
        self.assertIsNone(self.conf.foo)
        self.conf.set_default('foo', 'bar')
        self.assertEqual('bar', self.conf.foo)
        self.conf.clear_default('foo')
        self.assertIsNone(self.conf.foo)

    def test_default_override(self):
        self.conf.register_opt(cfg.StrOpt('foo', default='foo'))
        self.conf([])
        self.assertEqual('foo', self.conf.foo)
        self.conf.set_default('foo', 'bar')
        self.assertEqual('bar', self.conf.foo)
        self.conf.clear_default('foo')
        self.assertEqual('foo', self.conf.foo)

    def test_set_default_not_in_choices(self):
        self.conf.register_group(cfg.OptGroup('f'))
        self.conf.register_cli_opt(cfg.StrOpt('oo', choices=('a', 'b')), group='f')
        self.assertRaises(ValueError, self.conf.set_default, 'oo', 'c', 'f')

    def test_wrong_type_default_override(self):
        self.conf.register_opt(cfg.IntOpt('foo', default=1))
        self.conf([])
        self.assertEqual(1, self.conf.foo)
        self.assertRaises(ValueError, self.conf.set_default, 'foo', 'not_really_a_int')

    def test_override(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        self.conf.set_override('foo', 'bar')
        self.conf([])
        self.assertEqual('bar', self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test_override_none(self):
        self.conf.register_opt(cfg.StrOpt('foo', default='foo'))
        self.conf([])
        self.assertEqual('foo', self.conf.foo)
        self.conf.set_override('foo', None)
        self.assertIsNone(self.conf.foo)
        self.conf.clear_override('foo')
        self.assertEqual('foo', self.conf.foo)

    def test_group_no_default_override(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo'), group='blaa')
        self.conf([])
        self.assertIsNone(self.conf.blaa.foo)
        self.conf.set_default('foo', 'bar', group='blaa')
        self.assertEqual('bar', self.conf.blaa.foo)
        self.conf.clear_default('foo', group='blaa')
        self.assertIsNone(self.conf.blaa.foo)

    def test_group_default_override(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo', default='foo'), group='blaa')
        self.conf([])
        self.assertEqual('foo', self.conf.blaa.foo)
        self.conf.set_default('foo', 'bar', group='blaa')
        self.assertEqual('bar', self.conf.blaa.foo)
        self.conf.clear_default('foo', group='blaa')
        self.assertEqual('foo', self.conf.blaa.foo)

    def test_group_override(self):
        self.conf.register_group(cfg.OptGroup('blaa'))
        self.conf.register_opt(cfg.StrOpt('foo'), group='blaa')
        self.assertIsNone(self.conf.blaa.foo)
        self.conf.set_override('foo', 'bar', group='blaa')
        self.conf([])
        self.assertEqual('bar', self.conf.blaa.foo)
        self.conf.clear_override('foo', group='blaa')
        self.assertIsNone(self.conf.blaa.foo)

    def test_cli_bool_default(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        self.conf.set_default('foo', True)
        self.assertTrue(self.conf.foo)
        self.conf([])
        self.assertTrue(self.conf.foo)
        self.conf.set_default('foo', False)
        self.assertFalse(self.conf.foo)
        self.conf.clear_default('foo')
        self.assertIsNone(self.conf.foo)

    def test_cli_bool_override(self):
        self.conf.register_cli_opt(cfg.BoolOpt('foo'))
        self.conf.set_override('foo', True)
        self.assertTrue(self.conf.foo)
        self.conf([])
        self.assertTrue(self.conf.foo)
        self.conf.set_override('foo', False)
        self.assertFalse(self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test__str_override(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        self.conf.set_override('foo', True)
        self.conf([])
        self.assertEqual('True', self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test__wrong_type_override(self):
        self.conf.register_opt(cfg.IntOpt('foo'))
        self.assertRaises(ValueError, self.conf.set_override, 'foo', 'not_really_a_int')

    def test_set_override_in_choices(self):
        self.conf.register_group(cfg.OptGroup('f'))
        self.conf.register_cli_opt(cfg.StrOpt('oo', choices=('a', 'b')), group='f')
        self.conf.set_override('oo', 'b', 'f')
        self.assertEqual('b', self.conf.f.oo)

    def test_set_override_not_in_choices(self):
        self.conf.register_group(cfg.OptGroup('f'))
        self.conf.register_cli_opt(cfg.StrOpt('oo', choices=('a', 'b')), group='f')
        self.assertRaises(ValueError, self.conf.set_override, 'oo', 'c', 'f')

    def test_bool_override(self):
        self.conf.register_opt(cfg.BoolOpt('foo'))
        self.conf.set_override('foo', 'True')
        self.conf([])
        self.assertTrue(self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test_int_override_with_None(self):
        self.conf.register_opt(cfg.IntOpt('foo'))
        self.conf.set_override('foo', None)
        self.conf([])
        self.assertIsNone(self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test_str_override_with_None(self):
        self.conf.register_opt(cfg.StrOpt('foo'))
        self.conf.set_override('foo', None)
        self.conf([])
        self.assertIsNone(self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)

    def test_List_override(self):
        self.conf.register_opt(cfg.ListOpt('foo'))
        self.conf.set_override('foo', ['aa', 'bb'])
        self.conf([])
        self.assertEqual(['aa', 'bb'], self.conf.foo)
        self.conf.clear_override('foo')
        self.assertIsNone(self.conf.foo)
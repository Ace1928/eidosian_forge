import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
class TestMutableSection(tests.TestCase):
    scenarios = [('mutable', {'get_section': lambda opts: config.MutableSection('myID', opts)})]

    def test_set(self):
        a_dict = dict(foo='bar')
        section = self.get_section(a_dict)
        section.set('foo', 'new_value')
        self.assertEqual('new_value', section.get('foo'))
        self.assertEqual('new_value', a_dict.get('foo'))
        self.assertTrue('foo' in section.orig)
        self.assertEqual('bar', section.orig.get('foo'))

    def test_set_preserve_original_once(self):
        a_dict = dict(foo='bar')
        section = self.get_section(a_dict)
        section.set('foo', 'first_value')
        section.set('foo', 'second_value')
        self.assertTrue('foo' in section.orig)
        self.assertEqual('bar', section.orig.get('foo'))

    def test_remove(self):
        a_dict = dict(foo='bar')
        section = self.get_section(a_dict)
        section.remove('foo')
        self.assertEqual(None, section.get('foo'))
        self.assertEqual('unknown', section.get('foo', 'unknown'))
        self.assertFalse('foo' in section.options)
        self.assertTrue('foo' in section.orig)
        self.assertEqual('bar', section.orig.get('foo'))

    def test_remove_new_option(self):
        a_dict = dict()
        section = self.get_section(a_dict)
        section.set('foo', 'bar')
        section.remove('foo')
        self.assertFalse('foo' in section.options)
        self.assertTrue('foo' in section.orig)
        self.assertEqual(config._NewlyCreatedOption, section.orig['foo'])
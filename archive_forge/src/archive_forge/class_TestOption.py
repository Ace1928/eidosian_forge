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
class TestOption(tests.TestCase):

    def test_default_value(self):
        opt = config.Option('foo', default='bar')
        self.assertEqual('bar', opt.get_default())

    def test_callable_default_value(self):

        def bar_as_unicode():
            return 'bar'
        opt = config.Option('foo', default=bar_as_unicode)
        self.assertEqual('bar', opt.get_default())

    def test_default_value_from_env(self):
        opt = config.Option('foo', default='bar', default_from_env=['FOO'])
        self.overrideEnv('FOO', 'quux')
        self.assertEqual('quux', opt.get_default())

    def test_first_default_value_from_env_wins(self):
        opt = config.Option('foo', default='bar', default_from_env=['NO_VALUE', 'FOO', 'BAZ'])
        self.overrideEnv('FOO', 'foo')
        self.overrideEnv('BAZ', 'baz')
        self.assertEqual('foo', opt.get_default())

    def test_not_supported_list_default_value(self):
        self.assertRaises(AssertionError, config.Option, 'foo', default=[1])

    def test_not_supported_object_default_value(self):
        self.assertRaises(AssertionError, config.Option, 'foo', default=object())

    def test_not_supported_callable_default_value_not_unicode(self):

        def bar_not_unicode():
            return b'bar'
        opt = config.Option('foo', default=bar_not_unicode)
        self.assertRaises(AssertionError, opt.get_default)

    def test_get_help_topic(self):
        opt = config.Option('foo')
        self.assertEqual('foo', opt.get_help_topic())
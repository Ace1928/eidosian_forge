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
class TestIniConfigOptionExpansion(tests.TestCase):
    """Test option expansion from the IniConfig level.

    What we really want here is to test the Config level, but the class being
    abstract as far as storing values is concerned, this can't be done
    properly (yet).
    """

    def get_config(self, string=None):
        if string is None:
            string = b''
        c = config.IniBasedConfig.from_string(string)
        return c

    def assertExpansion(self, expected, conf, string, env=None):
        self.assertEqual(expected, conf.expand_options(string, env))

    def test_no_expansion(self):
        c = self.get_config('')
        self.assertExpansion('foo', c, 'foo')

    def test_env_adding_options(self):
        c = self.get_config('')
        self.assertExpansion('bar', c, '{foo}', {'foo': 'bar'})

    def test_env_overriding_options(self):
        c = self.get_config('foo=baz')
        self.assertExpansion('bar', c, '{foo}', {'foo': 'bar'})

    def test_simple_ref(self):
        c = self.get_config('foo=xxx')
        self.assertExpansion('xxx', c, '{foo}')

    def test_unknown_ref(self):
        c = self.get_config('')
        self.assertRaises(config.ExpandingUnknownOption, c.expand_options, '{foo}')

    def test_indirect_ref(self):
        c = self.get_config('\nfoo=xxx\nbar={foo}\n')
        self.assertExpansion('xxx', c, '{bar}')

    def test_embedded_ref(self):
        c = self.get_config('\nfoo=xxx\nbar=foo\n')
        self.assertExpansion('xxx', c, '{{bar}}')

    def test_simple_loop(self):
        c = self.get_config('foo={foo}')
        self.assertRaises(config.OptionExpansionLoop, c.expand_options, '{foo}')

    def test_indirect_loop(self):
        c = self.get_config('\nfoo={bar}\nbar={baz}\nbaz={foo}')
        e = self.assertRaises(config.OptionExpansionLoop, c.expand_options, '{foo}')
        self.assertEqual('foo->bar->baz', e.refs)
        self.assertEqual('{foo}', e.string)

    def test_list(self):
        conf = self.get_config('\nfoo=start\nbar=middle\nbaz=end\nlist={foo},{bar},{baz}\n')
        self.assertEqual(['start', 'middle', 'end'], conf.get_user_option('list', expand=True))

    def test_cascading_list(self):
        conf = self.get_config('\nfoo=start,{bar}\nbar=middle,{baz}\nbaz=end\nlist={foo}\n')
        self.assertEqual(['start', 'middle', 'end'], conf.get_user_option('list', expand=True))

    def test_pathological_hidden_list(self):
        conf = self.get_config('\nfoo=bin\nbar=go\nstart={foo\nmiddle=},{\nend=bar}\nhidden={start}{middle}{end}\n')
        self.assertEqual(['{foo', '}', '{', 'bar}'], conf.get_user_option('hidden', expand=True))
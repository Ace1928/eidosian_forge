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
class TestStackSet(TestStackWithTransport):

    def test_simple_set(self):
        conf = self.get_stack(self)
        self.assertEqual(None, conf.get('foo'))
        conf.set('foo', 'baz')
        self.assertEqual('baz', conf.get('foo'))

    def test_set_creates_a_new_section(self):
        conf = self.get_stack(self)
        conf.set('foo', 'baz')
        (self.assertEqual, 'baz', conf.get('foo'))

    def test_set_hook(self):
        calls = []

        def hook(*args):
            calls.append(args)
        config.ConfigHooks.install_named_hook('set', hook, None)
        self.assertLength(0, calls)
        conf = self.get_stack(self)
        conf.set('foo', 'bar')
        self.assertLength(1, calls)
        self.assertEqual((conf, 'foo', 'bar'), calls[0])
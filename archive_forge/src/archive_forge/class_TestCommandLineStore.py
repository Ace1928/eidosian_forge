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
class TestCommandLineStore(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.store = config.CommandLineStore()
        self.overrideAttr(config, 'option_registry', config.OptionRegistry())

    def get_section(self):
        """Get the unique section for the command line overrides."""
        sections = list(self.store.get_sections())
        self.assertLength(1, sections)
        store, section = sections[0]
        self.assertEqual(self.store, store)
        return section

    def test_no_override(self):
        self.store._from_cmdline([])
        section = self.get_section()
        self.assertLength(0, list(section.iter_option_names()))

    def test_simple_override(self):
        self.store._from_cmdline(['a=b'])
        section = self.get_section()
        self.assertEqual('b', section.get('a'))

    def test_list_override(self):
        opt = config.ListOption('l')
        config.option_registry.register(opt)
        self.store._from_cmdline(['l=1,2,3'])
        val = self.get_section().get('l')
        self.assertEqual('1,2,3', val)
        self.assertEqual(['1', '2', '3'], opt.convert_from_unicode(self.store, val))

    def test_multiple_overrides(self):
        self.store._from_cmdline(['a=b', 'x=y'])
        section = self.get_section()
        self.assertEqual('b', section.get('a'))
        self.assertEqual('y', section.get('x'))

    def test_wrong_syntax(self):
        self.assertRaises(errors.CommandError, self.store._from_cmdline, ['a=b', 'c'])
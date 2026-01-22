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
class TestTreeConfig(tests.TestCaseWithTransport):

    def test_get_value(self):
        """Test that retreiving a value from a section is possible"""
        branch = self.make_branch('.')
        tree_config = config.TreeConfig(branch)
        tree_config.set_option('value', 'key', 'SECTION')
        tree_config.set_option('value2', 'key2')
        tree_config.set_option('value3-top', 'key3')
        tree_config.set_option('value3-section', 'key3', 'SECTION')
        value = tree_config.get_option('key', 'SECTION')
        self.assertEqual(value, 'value')
        value = tree_config.get_option('key2')
        self.assertEqual(value, 'value2')
        self.assertEqual(tree_config.get_option('non-existant'), None)
        value = tree_config.get_option('non-existant', 'SECTION')
        self.assertEqual(value, None)
        value = tree_config.get_option('non-existant', default='default')
        self.assertEqual(value, 'default')
        self.assertEqual(tree_config.get_option('key2', 'NOSECTION'), None)
        value = tree_config.get_option('key2', 'NOSECTION', default='default')
        self.assertEqual(value, 'default')
        value = tree_config.get_option('key3')
        self.assertEqual(value, 'value3-top')
        value = tree_config.get_option('key3', 'SECTION')
        self.assertEqual(value, 'value3-section')
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
class TestStackCrossSectionsExpand(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()

    def get_config(self, location, string):
        if string is None:
            string = b''
        c = config.LocationStack(location)
        c.store._load_from_string(string)
        return c

    def test_dont_cross_unrelated_section(self):
        c = self.get_config('/another/branch/path', b'\n[/one/branch/path]\nfoo = hello\nbar = {foo}/2\n\n[/another/branch/path]\nbar = {foo}/2\n')
        self.assertRaises(config.ExpandingUnknownOption, c.get, 'bar', expand=True)

    def test_cross_related_sections(self):
        c = self.get_config('/project/branch/path', b'\n[/project]\nfoo = qu\n\n[/project/branch/path]\nbar = {foo}ux\n')
        self.assertEqual('quux', c.get('bar', expand=True))
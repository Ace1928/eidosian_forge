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
class TestRegisteredOptions(tests.TestCase):
    """All registered options should verify some constraints."""
    scenarios = [(key, {'option_name': key, 'option': option}) for key, option in config.option_registry.iteritems()]

    def setUp(self):
        super().setUp()
        self.registry = config.option_registry

    def test_proper_name(self):
        self.assertEqual(self.option_name, self.option.name)

    def test_help_is_set(self):
        option_help = self.registry.get_help(self.option_name)
        self.assertIsNot(None, option_help)
        self.assertNotEqual('', option_help)
import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_optional_extensions(self):
    self.assertIsInstance(self.optional_extensions, list)
    if not self.optional_extensions:
        self.skipTest('API definition has no optional extensions.')
    for ext in self.optional_extensions:
        self.assertIn(ext, base.KNOWN_EXTENSIONS, 'Optional extension is unknown, check for typos.')
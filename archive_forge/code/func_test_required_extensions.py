import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_required_extensions(self):
    self.assertIsInstance(self.required_extensions, list)
    if not self.required_extensions:
        self.skipTest('API definition has no required extensions.')
    for ext in self.required_extensions:
        self.assertIn(ext, base.KNOWN_EXTENSIONS, 'Required extension is unknown, check for typos.')
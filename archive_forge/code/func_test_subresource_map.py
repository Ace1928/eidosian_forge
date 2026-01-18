import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_subresource_map(self):
    if not self.subresource_map:
        self.skipTest('API extension has no subresource map.')
    for subresource in self.subresource_map:
        self.assertIn(subresource, self.extension_subresources + base.KNOWN_RESOURCES, 'Sub-resource is unknown, check for typos.')
        sub_attrmap = self.subresource_map[subresource]
        if 'parent' in sub_attrmap:
            self.assertEqual(2, len(sub_attrmap.keys()))
            self.assertIn('parent', sub_attrmap)
            self.assertIn('parameters', sub_attrmap)
            self._assert_subresource(subresource)
        else:
            self.assertEqual(['parameters'], [p for p in sub_attrmap.keys()], 'When extending sub-resources only use the parameters keyword')
            self.assertParams(sub_attrmap['parameters'])
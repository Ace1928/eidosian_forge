import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_list(self):
    self.resource_provider_create()
    retrieved = self.resource_provider_list()[0]
    self.assertIn('root_provider_uuid', retrieved)
    self.assertIn('parent_provider_uuid', retrieved)
import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_show(self):
    created = self.resource_provider_create()
    retrieved = self.resource_provider_show(created['uuid'])
    self.assertIn('root_provider_uuid', retrieved)
    self.assertIn('parent_provider_uuid', retrieved)
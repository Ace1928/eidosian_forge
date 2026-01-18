import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_create(self):
    created = self.resource_provider_create()
    self.assertIn('root_provider_uuid', created)
    self.assertIn('parent_provider_uuid', created)
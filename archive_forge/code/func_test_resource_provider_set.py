import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_set(self):
    created = self.resource_provider_create()
    updated = self.resource_provider_set(created['uuid'], name='some_new_name')
    self.assertIn('root_provider_uuid', updated)
    self.assertIn('parent_provider_uuid', updated)
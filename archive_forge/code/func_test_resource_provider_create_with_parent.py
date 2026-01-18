import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_create_with_parent(self):
    parent = self.resource_provider_create()
    child = self.resource_provider_create(parent_provider_uuid=parent['uuid'])
    self.assertEqual(child['parent_provider_uuid'], parent['uuid'])
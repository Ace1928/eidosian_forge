import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_delete_parent(self):
    parent = self.resource_provider_create()
    self.resource_provider_create(parent_provider_uuid=parent['uuid'])
    exc = self.assertRaises(base.CommandException, self.resource_provider_delete, parent['uuid'])
    self.assertIn('HTTP 409', str(exc))
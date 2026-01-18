import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_delete(self):
    created = self.resource_provider_create()
    before_delete = self.resource_provider_list(uuid=created['uuid'])
    self.assertEqual([created['uuid']], [rp['uuid'] for rp in before_delete])
    self.resource_provider_delete(created['uuid'])
    after_delete = self.resource_provider_list(uuid=created['uuid'])
    self.assertEqual([], after_delete)
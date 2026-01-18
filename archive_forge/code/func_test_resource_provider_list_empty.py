import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_list_empty(self):
    by_name = self.resource_provider_list(name='some_non_existing_name')
    self.assertEqual([], by_name)
    by_uuid = self.resource_provider_list(uuid=str(uuid.uuid4()))
    self.assertEqual([], by_uuid)
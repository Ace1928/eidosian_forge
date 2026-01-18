import operator
import uuid
from osc_placement.tests.functional import base
def test_resource_provider_list_by_uuid(self):
    rp1 = self.resource_provider_create()
    self.resource_provider_create()
    expected_filtered_by_uuid = [rp1]
    self.assertEqual(expected_filtered_by_uuid, [rp for rp in self.resource_provider_list(uuid=rp1['uuid'])])
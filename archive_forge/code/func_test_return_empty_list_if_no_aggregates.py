import uuid
from osc_placement.tests.functional import base
def test_return_empty_list_if_no_aggregates(self):
    rp = self.resource_provider_create()
    self.assertEqual([], self.resource_provider_aggregate_list(rp['uuid']))
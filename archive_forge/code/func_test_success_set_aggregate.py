import uuid
from osc_placement.tests.functional import base
def test_success_set_aggregate(self):
    rp = self.resource_provider_create()
    aggs = {str(uuid.uuid4()) for _ in range(2)}
    rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs, generation=rp['generation'])
    self.assertEqual(aggs, {r['uuid'] for r in rows})
    rows = self.resource_provider_aggregate_list(rp['uuid'])
    self.assertEqual(aggs, {r['uuid'] for r in rows})
    self.resource_provider_aggregate_set(rp['uuid'], *[], generation=rp['generation'] + 1)
    rows = self.resource_provider_aggregate_list(rp['uuid'])
    self.assertEqual([], rows)
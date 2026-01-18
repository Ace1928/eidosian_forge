import uuid
from osc_placement.tests.functional import base
def test_success_set_large_number_aggregates(self):
    rp = self.resource_provider_create()
    aggs = {str(uuid.uuid4()) for _ in range(100)}
    rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs, generation=rp['generation'])
    self.assertEqual(aggs, {r['uuid'] for r in rows})
    rows = self.resource_provider_aggregate_set(rp['uuid'], *[], generation=rp['generation'] + 1)
    self.assertEqual([], rows)
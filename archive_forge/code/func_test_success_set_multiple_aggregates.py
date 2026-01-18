import uuid
from osc_placement.tests.functional import base
def test_success_set_multiple_aggregates(self):
    rps = [self.resource_provider_create() for _ in range(2)]
    aggs = {str(uuid.uuid4()) for _ in range(2)}
    for rp in rps:
        rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs, generation=rp['generation'])
        self.assertEqual(aggs, {r['uuid'] for r in rows})
    rows = self.resource_provider_aggregate_set(rps[0]['uuid'], *[], generation=rp['generation'] + 1)
    self.assertEqual([], rows)
    rows = self.resource_provider_aggregate_list(rps[1]['uuid'])
    self.assertEqual(aggs, {r['uuid'] for r in rows})
    rows = self.resource_provider_aggregate_set(rps[1]['uuid'], *[], generation=rp['generation'] + 1)
    self.assertEqual([], rows)
import uuid
from osc_placement.tests.functional import base
class TestAggregate119(TestAggregate):
    VERSION = '1.19'

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

    def test_success_set_large_number_aggregates(self):
        rp = self.resource_provider_create()
        aggs = {str(uuid.uuid4()) for _ in range(100)}
        rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs, generation=rp['generation'])
        self.assertEqual(aggs, {r['uuid'] for r in rows})
        rows = self.resource_provider_aggregate_set(rp['uuid'], *[], generation=rp['generation'] + 1)
        self.assertEqual([], rows)

    def test_fail_incorrect_generation(self):
        rp = self.resource_provider_create()
        agg = str(uuid.uuid4())
        self.assertCommandFailed('Please update the generation and try again.', self.resource_provider_aggregate_set, rp['uuid'], agg, generation=99999)

    def test_fail_generation_not_int(self):
        rp = self.resource_provider_create()
        agg = str(uuid.uuid4())
        self.assertCommandFailed('invalid int value', self.resource_provider_aggregate_set, rp['uuid'], agg, generation='barney')

    def test_fail_if_incorrect_aggregate_uuid(self):
        rp = self.resource_provider_create()
        aggs = ['abc', 'efg']
        self.assertCommandFailed("is not a 'uuid'", self.resource_provider_aggregate_set, rp['uuid'], *aggs, generation=rp['generation'])

    def test_fail_generation_arg_version_handling(self):
        rp = self.resource_provider_create()
        agg = str(uuid.uuid4())
        self.assertCommandFailed('A generation must be specified.', self.resource_provider_aggregate_set, rp['uuid'], agg)
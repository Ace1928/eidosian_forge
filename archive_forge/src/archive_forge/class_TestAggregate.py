import uuid
from osc_placement.tests.functional import base
class TestAggregate(base.BaseTestCase):
    VERSION = '1.1'

    def test_fail_if_no_rp(self):
        self.assertCommandFailed(base.ARGUMENTS_MISSING, self.openstack, 'resource provider aggregate list')

    def test_fail_if_rp_not_found(self):
        self.assertCommandFailed('No resource provider', self.resource_provider_aggregate_list, 'fake-uuid')

    def test_return_empty_list_if_no_aggregates(self):
        rp = self.resource_provider_create()
        self.assertEqual([], self.resource_provider_aggregate_list(rp['uuid']))

    def test_success_set_aggregate(self):
        rp = self.resource_provider_create()
        aggs = {str(uuid.uuid4()) for _ in range(2)}
        rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs)
        self.assertEqual(aggs, {r['uuid'] for r in rows})
        rows = self.resource_provider_aggregate_list(rp['uuid'])
        self.assertEqual(aggs, {r['uuid'] for r in rows})
        self.resource_provider_aggregate_set(rp['uuid'])
        rows = self.resource_provider_aggregate_list(rp['uuid'])
        self.assertEqual([], rows)

    def test_set_aggregate_fail_if_no_rp(self):
        self.assertCommandFailed(base.ARGUMENTS_MISSING, self.openstack, 'resource provider aggregate set')

    def test_success_set_multiple_aggregates(self):
        rps = [self.resource_provider_create() for _ in range(2)]
        aggs = {str(uuid.uuid4()) for _ in range(2)}
        for rp in rps:
            rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs)
            self.assertEqual(aggs, {r['uuid'] for r in rows})
        rows = self.resource_provider_aggregate_set(rps[0]['uuid'])
        self.assertEqual([], rows)
        rows = self.resource_provider_aggregate_list(rps[1]['uuid'])
        self.assertEqual(aggs, {r['uuid'] for r in rows})
        rows = self.resource_provider_aggregate_set(rps[1]['uuid'])
        self.assertEqual([], rows)

    def test_success_set_large_number_aggregates(self):
        rp = self.resource_provider_create()
        aggs = {str(uuid.uuid4()) for _ in range(100)}
        rows = self.resource_provider_aggregate_set(rp['uuid'], *aggs)
        self.assertEqual(aggs, {r['uuid'] for r in rows})
        rows = self.resource_provider_aggregate_set(rp['uuid'])
        self.assertEqual([], rows)

    def test_fail_if_incorrect_aggregate_uuid(self):
        rp = self.resource_provider_create()
        aggs = ['abc', 'efg']
        self.assertCommandFailed("is not a 'uuid'", self.resource_provider_aggregate_set, rp['uuid'], *aggs)

    def test_fail_generation_arg_version_handling(self):
        rp = self.resource_provider_create()
        agg = str(uuid.uuid4())
        self.assertCommandFailed('Operation or argument is not supported with version 1.1', self.resource_provider_aggregate_set, rp['uuid'], agg, generation=5)
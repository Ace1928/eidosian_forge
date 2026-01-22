from openstack.identity.v3 import registered_limit
from openstack.tests.unit import base
class TestRegistered_limit(base.TestCase):

    def test_basic(self):
        sot = registered_limit.RegisteredLimit()
        self.assertEqual('registered_limit', sot.resource_key)
        self.assertEqual('registered_limits', sot.resources_key)
        self.assertEqual('/registered_limits', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)
        self.assertDictEqual({'service_id': 'service_id', 'region_id': 'region_id', 'resource_name': 'resource_name', 'marker': 'marker', 'limit': 'limit'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = registered_limit.RegisteredLimit(**EXAMPLE)
        self.assertEqual(EXAMPLE['service_id'], sot.service_id)
        self.assertEqual(EXAMPLE['region_id'], sot.region_id)
        self.assertEqual(EXAMPLE['resource_name'], sot.resource_name)
        self.assertEqual(EXAMPLE['default_limit'], sot.default_limit)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['links'], sot.links)
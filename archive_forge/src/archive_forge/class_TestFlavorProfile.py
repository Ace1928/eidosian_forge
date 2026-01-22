import uuid
from openstack.load_balancer.v2 import flavor_profile
from openstack.tests.unit import base
class TestFlavorProfile(base.TestCase):

    def test_basic(self):
        test_profile = flavor_profile.FlavorProfile()
        self.assertEqual('flavorprofile', test_profile.resource_key)
        self.assertEqual('flavorprofiles', test_profile.resources_key)
        self.assertEqual('/lbaas/flavorprofiles', test_profile.base_path)
        self.assertTrue(test_profile.allow_create)
        self.assertTrue(test_profile.allow_fetch)
        self.assertTrue(test_profile.allow_commit)
        self.assertTrue(test_profile.allow_delete)
        self.assertTrue(test_profile.allow_list)

    def test_make_it(self):
        test_profile = flavor_profile.FlavorProfile(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], test_profile.id)
        self.assertEqual(EXAMPLE['name'], test_profile.name)
        self.assertEqual(EXAMPLE['provider_name'], test_profile.provider_name)
        self.assertEqual(EXAMPLE['flavor_data'], test_profile.flavor_data)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'id': 'id', 'name': 'name', 'provider_name': 'provider_name', 'flavor_data': 'flavor_data'}, test_profile._query_mapping._mapping)
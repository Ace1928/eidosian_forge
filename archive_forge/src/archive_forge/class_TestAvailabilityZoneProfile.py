import uuid
from openstack.load_balancer.v2 import availability_zone_profile
from openstack.tests.unit import base
class TestAvailabilityZoneProfile(base.TestCase):

    def test_basic(self):
        test_profile = availability_zone_profile.AvailabilityZoneProfile()
        self.assertEqual('availability_zone_profile', test_profile.resource_key)
        self.assertEqual('availability_zone_profiles', test_profile.resources_key)
        self.assertEqual('/lbaas/availabilityzoneprofiles', test_profile.base_path)
        self.assertTrue(test_profile.allow_create)
        self.assertTrue(test_profile.allow_fetch)
        self.assertTrue(test_profile.allow_commit)
        self.assertTrue(test_profile.allow_delete)
        self.assertTrue(test_profile.allow_list)

    def test_make_it(self):
        test_profile = availability_zone_profile.AvailabilityZoneProfile(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], test_profile.id)
        self.assertEqual(EXAMPLE['name'], test_profile.name)
        self.assertEqual(EXAMPLE['provider_name'], test_profile.provider_name)
        self.assertEqual(EXAMPLE['availability_zone_data'], test_profile.availability_zone_data)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'id': 'id', 'name': 'name', 'provider_name': 'provider_name', 'availability_zone_data': 'availability_zone_data'}, test_profile._query_mapping._mapping)
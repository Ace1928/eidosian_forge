from openstack.network.v2 import service_profile
from openstack.tests.unit import base
class TestServiceProfile(base.TestCase):

    def test_basic(self):
        service_profiles = service_profile.ServiceProfile()
        self.assertEqual('service_profile', service_profiles.resource_key)
        self.assertEqual('service_profiles', service_profiles.resources_key)
        self.assertEqual('/service_profiles', service_profiles.base_path)
        self.assertTrue(service_profiles.allow_create)
        self.assertTrue(service_profiles.allow_fetch)
        self.assertTrue(service_profiles.allow_commit)
        self.assertTrue(service_profiles.allow_delete)
        self.assertTrue(service_profiles.allow_list)

    def test_make_it(self):
        service_profiles = service_profile.ServiceProfile(**EXAMPLE)
        self.assertEqual(EXAMPLE['driver'], service_profiles.driver)

    def test_make_it_with_optional(self):
        service_profiles = service_profile.ServiceProfile(**EXAMPLE_WITH_OPTIONAL)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['description'], service_profiles.description)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['driver'], service_profiles.driver)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['enabled'], service_profiles.is_enabled)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['metainfo'], service_profiles.meta_info)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['project_id'], service_profiles.project_id)
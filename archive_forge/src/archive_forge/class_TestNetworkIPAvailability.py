from openstack.network.v2 import network_ip_availability
from openstack.tests.unit import base
class TestNetworkIPAvailability(base.TestCase):

    def test_basic(self):
        sot = network_ip_availability.NetworkIPAvailability()
        self.assertEqual('network_ip_availability', sot.resource_key)
        self.assertEqual('network_ip_availabilities', sot.resources_key)
        self.assertEqual('/network-ip-availabilities', sot.base_path)
        self.assertEqual('network_name', sot.name_attribute)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = network_ip_availability.NetworkIPAvailability(**EXAMPLE)
        self.assertEqual(EXAMPLE['network_id'], sot.network_id)
        self.assertEqual(EXAMPLE['network_name'], sot.network_name)
        self.assertEqual(EXAMPLE['subnet_ip_availability'], sot.subnet_ip_availability)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['total_ips'], sot.total_ips)
        self.assertEqual(EXAMPLE['used_ips'], sot.used_ips)

    def test_make_it_with_optional(self):
        sot = network_ip_availability.NetworkIPAvailability(**EXAMPLE_WITH_OPTIONAL)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['network_id'], sot.network_id)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['network_name'], sot.network_name)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['subnet_ip_availability'], sot.subnet_ip_availability)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['total_ips'], sot.total_ips)
        self.assertEqual(EXAMPLE_WITH_OPTIONAL['used_ips'], sot.used_ips)
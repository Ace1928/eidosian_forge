from openstack.tests import fakes
from openstack.tests.unit import base
def test_search_floating_ips(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep})])
    floating_ips = self.cloud.search_floating_ips(filters={'attached': False})
    self.assertIsInstance(floating_ips, list)
    self.assertEqual(2, len(floating_ips))
    self.assertAreInstances(floating_ips, dict)
    self.assert_calls()
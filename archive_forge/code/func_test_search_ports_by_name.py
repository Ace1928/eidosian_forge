from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_search_ports_by_name(self):
    port_name = 'first-port'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports']), json=self.mock_neutron_port_list_rep)])
    ports = self.cloud.search_ports(name_or_id=port_name)
    self.assertEqual(1, len(ports))
    self.assertEqual('fa:16:3e:58:42:ed', ports[0]['mac_address'])
    self.assert_calls()
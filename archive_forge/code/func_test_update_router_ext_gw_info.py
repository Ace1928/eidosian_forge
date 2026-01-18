import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_update_router_ext_gw_info(self):
    router = self._create_and_verify_advanced_router(external_cidr=u'10.9.9.0/24')
    existing_net_id = router['external_gateway_info']['network_id']
    sub_name = self.subnet_prefix + '_update'
    sub = self.operator_cloud.create_subnet(existing_net_id, '10.10.10.0/24', subnet_name=sub_name, gateway_ip='10.10.10.1')
    updated = self.operator_cloud.update_router(router['id'], ext_gateway_net_id=existing_net_id, ext_fixed_ips=[{'subnet_id': sub['id'], 'ip_address': '10.10.10.77'}])
    self.assertIsNotNone(updated)
    for field in EXPECTED_TOPLEVEL_FIELDS:
        self.assertIn(field, updated)
    ext_gw_info = updated['external_gateway_info']
    self.assertEqual(1, len(ext_gw_info['external_fixed_ips']))
    self.assertEqual(sub['id'], ext_gw_info['external_fixed_ips'][0]['subnet_id'])
    self.assertEqual('10.10.10.77', ext_gw_info['external_fixed_ips'][0]['ip_address'])
    self.assertEqual(router['status'], updated['status'])
    self.assertEqual(router['name'], updated['name'])
    self.assertEqual(router['admin_state_up'], updated['admin_state_up'])
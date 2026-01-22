import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def _create_and_verify_advanced_router(self, external_cidr, external_gateway_ip=None):
    net1_name = self.network_prefix + '_net1'
    sub1_name = self.subnet_prefix + '_sub1'
    net1 = self.operator_cloud.create_network(name=net1_name, external=True)
    sub1 = self.operator_cloud.create_subnet(net1['id'], external_cidr, subnet_name=sub1_name, gateway_ip=external_gateway_ip)
    ip_net = ipaddress.IPv4Network(external_cidr)
    last_ip = str(list(ip_net.hosts())[-1])
    router_name = self.router_prefix + '_create_advanced'
    router = self.operator_cloud.create_router(name=router_name, admin_state_up=False, ext_gateway_net_id=net1['id'], enable_snat=False, ext_fixed_ips=[{'subnet_id': sub1['id'], 'ip_address': last_ip}])
    for field in EXPECTED_TOPLEVEL_FIELDS:
        self.assertIn(field, router)
    ext_gw_info = router['external_gateway_info']
    for field in EXPECTED_GW_INFO_FIELDS:
        self.assertIn(field, ext_gw_info)
    self.assertEqual(router_name, router['name'])
    self.assertEqual('ACTIVE', router['status'])
    self.assertFalse(router['admin_state_up'])
    self.assertEqual(1, len(ext_gw_info['external_fixed_ips']))
    self.assertEqual(sub1['id'], ext_gw_info['external_fixed_ips'][0]['subnet_id'])
    self.assertEqual(last_ip, ext_gw_info['external_fixed_ips'][0]['ip_address'])
    return router
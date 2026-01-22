from openstackclient.tests.functional.network.v2 import common
def _create_ndp_proxies(self, ndp_proxies):
    for ndp_proxy in ndp_proxies:
        output = self.openstack('router ndp proxy create %(router)s --name %(name)s --port %(port)s --ip-address %(address)s' % {'router': ndp_proxy['router_id'], 'name': ndp_proxy['name'], 'port': ndp_proxy['port_id'], 'address': ndp_proxy['address']}, parse_output=True)
        self.assertEqual(ndp_proxy['router_id'], output['router_id'])
        self.assertEqual(ndp_proxy['port_id'], output['port_id'])
        self.assertEqual(ndp_proxy['address'], output['ip_address'])
        self.created_ndp_proxies.append(output)
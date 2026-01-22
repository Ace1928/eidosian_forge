from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
class TestDescribeVPNConnections(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return DESCRIBE_VPNCONNECTIONS

    def test_get_vpcs(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_vpn_connections(['vpn-12qw34er56ty', 'vpn-qwerty12'], filters=[('state', ['pending', 'available'])])
        self.assert_request_parameters({'Action': 'DescribeVpnConnections', 'VpnConnectionId.1': 'vpn-12qw34er56ty', 'VpnConnectionId.2': 'vpn-qwerty12', 'Filter.1.Name': 'state', 'Filter.1.Value.1': 'pending', 'Filter.1.Value.2': 'available'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(len(api_response), 2)
        vpn0 = api_response[0]
        self.assertEqual(vpn0.type, 'ipsec.1')
        self.assertEqual(vpn0.customer_gateway_id, 'cgw-1234qwe9')
        self.assertEqual(vpn0.vpn_gateway_id, 'vgw-lkjh1234')
        self.assertEqual(len(vpn0.tunnels), 2)
        self.assertDictEqual(vpn0.tags, {'Name': 'VPN 1'})
        vpn1 = api_response[1]
        self.assertEqual(vpn1.state, 'pending')
        self.assertEqual(len(vpn1.static_routes), 1)
        self.assertTrue(vpn1.options.static_routes_only)
        self.assertEqual(vpn1.tunnels[0].status, 'UP')
        self.assertEqual(vpn1.tunnels[1].status, 'UP')
        self.assertDictEqual(vpn1.tags, {})
        self.assertEqual(vpn1.static_routes[0].source, 'static')
        self.assertEqual(vpn1.static_routes[0].state, 'pending')
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
class TestCreateVPNConnection(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateVpnConnectionResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>5cc7891f-1f3b-4fc4-a626-bdea8f63ff5a</requestId>\n              <vpnConnection>\n                <vpnConnectionId>vpn-83ad48ea</vpnConnectionId>\n                <state>pending</state>\n                <customerGatewayConfiguration>\n                    &lt;?xml version="1.0" encoding="UTF-8"?&gt;\n                </customerGatewayConfiguration>\n                <customerGatewayId>cgw-b4dc3961</customerGatewayId>\n                <vpnGatewayId>vgw-8db04f81</vpnGatewayId>\n                <options>\n                  <staticRoutesOnly>true</staticRoutesOnly>\n                </options>\n                <routes/>\n              </vpnConnection>\n            </CreateVpnConnectionResponse>\n        '

    def test_create_vpn_connection(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_vpn_connection('ipsec.1', 'cgw-b4dc3961', 'vgw-8db04f81', static_routes_only=True)
        self.assert_request_parameters({'Action': 'CreateVpnConnection', 'Type': 'ipsec.1', 'CustomerGatewayId': 'cgw-b4dc3961', 'VpnGatewayId': 'vgw-8db04f81', 'Options.StaticRoutesOnly': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, VpnConnection)
        self.assertEquals(api_response.id, 'vpn-83ad48ea')
        self.assertEquals(api_response.customer_gateway_id, 'cgw-b4dc3961')
        self.assertEquals(api_response.options.static_routes_only, True)
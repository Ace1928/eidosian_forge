from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
class TestCreateVpnGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateVpnGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <vpnGateway>\n                <vpnGatewayId>vgw-8db04f81</vpnGatewayId>\n                <state>pending</state>\n                <type>ipsec.1</type>\n                <availabilityZone>us-east-1a</availabilityZone>\n                <attachments/>\n                <tagSet/>\n              </vpnGateway>\n            </CreateVpnGatewayResponse>\n        '

    def test_delete_vpn_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_vpn_gateway('ipsec.1', 'us-east-1a')
        self.assert_request_parameters({'Action': 'CreateVpnGateway', 'AvailabilityZone': 'us-east-1a', 'Type': 'ipsec.1'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, VpnGateway)
        self.assertEquals(api_response.id, 'vgw-8db04f81')
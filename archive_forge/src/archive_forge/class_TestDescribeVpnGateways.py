from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
class TestDescribeVpnGateways(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DescribeVpnGatewaysResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n              <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n              <vpnGatewaySet>\n                <item>\n                  <vpnGatewayId>vgw-8db04f81</vpnGatewayId>\n                  <state>available</state>\n                  <type>ipsec.1</type>\n                  <availabilityZone>us-east-1a</availabilityZone>\n                  <attachments>\n                    <item>\n                      <vpcId>vpc-1a2b3c4d</vpcId>\n                      <state>attached</state>\n                    </item>\n                  </attachments>\n                  <tagSet/>\n                </item>\n              </vpnGatewaySet>\n            </DescribeVpnGatewaysResponse>\n        '

    def test_get_all_vpn_gateways(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_vpn_gateways('vgw-8db04f81', filters=OrderedDict([('state', ['pending', 'available']), ('availability-zone', 'us-east-1a')]))
        self.assert_request_parameters({'Action': 'DescribeVpnGateways', 'VpnGatewayId.1': 'vgw-8db04f81', 'Filter.1.Name': 'state', 'Filter.1.Value.1': 'pending', 'Filter.1.Value.2': 'available', 'Filter.2.Name': 'availability-zone', 'Filter.2.Value.1': 'us-east-1a'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEqual(len(api_response), 1)
        self.assertIsInstance(api_response[0], VpnGateway)
        self.assertEqual(api_response[0].id, 'vgw-8db04f81')
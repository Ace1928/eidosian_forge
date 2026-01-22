from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
class TestCreateCustomerGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateCustomerGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <customerGateway>\n                  <customerGatewayId>cgw-b4dc3961</customerGatewayId>\n                  <state>pending</state>\n                  <type>ipsec.1</type>\n                  <ipAddress>12.1.2.3</ipAddress>\n                  <bgpAsn>65534</bgpAsn>\n                  <tagSet/>\n               </customerGateway>\n            </CreateCustomerGatewayResponse>\n        '

    def test_create_customer_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_customer_gateway('ipsec.1', '12.1.2.3', 65534)
        self.assert_request_parameters({'Action': 'CreateCustomerGateway', 'Type': 'ipsec.1', 'IpAddress': '12.1.2.3', 'BgpAsn': 65534}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, CustomerGateway)
        self.assertEquals(api_response.id, 'cgw-b4dc3961')
        self.assertEquals(api_response.state, 'pending')
        self.assertEquals(api_response.type, 'ipsec.1')
        self.assertEquals(api_response.ip_address, '12.1.2.3')
        self.assertEquals(api_response.bgp_asn, 65534)
from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
class TestDeleteCustomerGateway(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <DeleteCustomerGatewayResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n               <return>true</return>\n            </DeleteCustomerGatewayResponse>\n        '

    def test_delete_customer_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.delete_customer_gateway('cgw-b4dc3961')
        self.assert_request_parameters({'Action': 'DeleteCustomerGateway', 'CustomerGatewayId': 'cgw-b4dc3961'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)
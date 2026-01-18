from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_replace_route_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.replace_route('rtb-e4ad488d', '0.0.0.0/0', gateway_id='igw-eaad4883')
    self.assert_request_parameters({'Action': 'ReplaceRoute', 'RouteTableId': 'rtb-e4ad488d', 'DestinationCidrBlock': '0.0.0.0/0', 'GatewayId': 'igw-eaad4883'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)
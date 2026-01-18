from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_replace_route_instance(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.replace_route('rtb-g8ff4ea2', '0.0.0.0/0', instance_id='i-1a2b3c4d')
    self.assert_request_parameters({'Action': 'ReplaceRoute', 'RouteTableId': 'rtb-g8ff4ea2', 'DestinationCidrBlock': '0.0.0.0/0', 'InstanceId': 'i-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)
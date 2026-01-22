from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestCreateRoute(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateRouteResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <return>true</return>\n            </CreateRouteResponse>\n        '

    def test_create_route_gateway(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_route('rtb-e4ad488d', '0.0.0.0/0', gateway_id='igw-eaad4883')
        self.assert_request_parameters({'Action': 'CreateRoute', 'RouteTableId': 'rtb-e4ad488d', 'DestinationCidrBlock': '0.0.0.0/0', 'GatewayId': 'igw-eaad4883'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)

    def test_create_route_instance(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_route('rtb-g8ff4ea2', '0.0.0.0/0', instance_id='i-1a2b3c4d')
        self.assert_request_parameters({'Action': 'CreateRoute', 'RouteTableId': 'rtb-g8ff4ea2', 'DestinationCidrBlock': '0.0.0.0/0', 'InstanceId': 'i-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)

    def test_create_route_interface(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_route('rtb-g8ff4ea2', '0.0.0.0/0', interface_id='eni-1a2b3c4d')
        self.assert_request_parameters({'Action': 'CreateRoute', 'RouteTableId': 'rtb-g8ff4ea2', 'DestinationCidrBlock': '0.0.0.0/0', 'NetworkInterfaceId': 'eni-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)

    def test_create_route_vpc_peering_connection(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_route('rtb-g8ff4ea2', '0.0.0.0/0', vpc_peering_connection_id='pcx-1a2b3c4d')
        self.assert_request_parameters({'Action': 'CreateRoute', 'RouteTableId': 'rtb-g8ff4ea2', 'DestinationCidrBlock': '0.0.0.0/0', 'VpcPeeringConnectionId': 'pcx-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertEquals(api_response, True)
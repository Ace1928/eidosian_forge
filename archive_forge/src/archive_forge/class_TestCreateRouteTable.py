from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
class TestCreateRouteTable(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n            <CreateRouteTableResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n               <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n               <routeTable>\n                  <routeTableId>rtb-f9ad4890</routeTableId>\n                  <vpcId>vpc-11ad4878</vpcId>\n                  <routeSet>\n                     <item>\n                        <destinationCidrBlock>10.0.0.0/22</destinationCidrBlock>\n                        <gatewayId>local</gatewayId>\n                        <state>active</state>\n                     </item>\n                  </routeSet>\n                  <associationSet/>\n                  <tagSet/>\n               </routeTable>\n            </CreateRouteTableResponse>\n        '

    def test_create_route_table(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_route_table('vpc-11ad4878')
        self.assert_request_parameters({'Action': 'CreateRouteTable', 'VpcId': 'vpc-11ad4878'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
        self.assertIsInstance(api_response, RouteTable)
        self.assertEquals(api_response.id, 'rtb-f9ad4890')
        self.assertEquals(len(api_response.routes), 1)
        self.assertEquals(api_response.routes[0].destination_cidr_block, '10.0.0.0/22')
        self.assertEquals(api_response.routes[0].gateway_id, 'local')
        self.assertEquals(api_response.routes[0].state, 'active')
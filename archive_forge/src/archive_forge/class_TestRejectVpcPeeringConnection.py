from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
class TestRejectVpcPeeringConnection(AWSMockServiceTestCase):
    REJECT_VPC_PEERING_CONNECTION = b'<RejectVpcPeeringConnectionResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n  <return>true</return>\n</RejectVpcPeeringConnectionResponse>'
    connection_class = VPCConnection

    def default_body(self):
        return self.REJECT_VPC_PEERING_CONNECTION

    def test_reject_vpc_peering_connection(self):
        self.set_http_response(status_code=200)
        self.assertEquals(self.service_connection.reject_vpc_peering_connection('pcx-12345678'), True)
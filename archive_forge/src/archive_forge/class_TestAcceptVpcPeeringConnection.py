from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
class TestAcceptVpcPeeringConnection(AWSMockServiceTestCase):
    ACCEPT_VPC_PEERING_CONNECTION = b'<AcceptVpcPeeringConnectionResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n  <vpcPeeringConnection>\n        <vpcPeeringConnectionId>pcx-1a2b3c4d</vpcPeeringConnectionId>\n       <requesterVpcInfo>\n            <ownerId>123456789012</ownerId>\n            <vpcId>vpc-1a2b3c4d</vpcId>\n            <cidrBlock>10.0.0.0/28</cidrBlock>\n        </requesterVpcInfo>\n        <accepterVpcInfo>\n            <ownerId>777788889999</ownerId>\n            <vpcId>vpc-111aaa22</vpcId>\n            <cidrBlock>10.0.1.0/28</cidrBlock>\n        </accepterVpcInfo>\n        <status>\n            <code>active</code>\n            <message>Active</message>\n        </status>\n        <tagSet/>\n    </vpcPeeringConnection>\n</AcceptVpcPeeringConnectionResponse>'
    connection_class = VPCConnection

    def default_body(self):
        return self.ACCEPT_VPC_PEERING_CONNECTION

    def test_accept_vpc_peering_connection(self):
        self.set_http_response(status_code=200)
        vpc_peering_connection = self.service_connection.accept_vpc_peering_connection('pcx-1a2b3c4d')
        self.assertEqual(vpc_peering_connection.id, 'pcx-1a2b3c4d')
        self.assertEqual(vpc_peering_connection.status_code, 'active')
        self.assertEqual(vpc_peering_connection.status_message, 'Active')
        self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '123456789012')
        self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-1a2b3c4d')
        self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '10.0.0.0/28')
        self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '777788889999')
        self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-111aaa22')
        self.assertEqual(vpc_peering_connection.accepter_vpc_info.cidr_block, '10.0.1.0/28')
from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
class TestDescribeVpcPeeringConnections(AWSMockServiceTestCase):
    DESCRIBE_VPC_PEERING_CONNECTIONS = b'<?xml version="1.0" encoding="UTF-8"?>\n<DescribeVpcPeeringConnectionsResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n   <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n    <vpcPeeringConnectionSet>\n        <item>\n           <vpcPeeringConnectionId>pcx-111aaa22</vpcPeeringConnectionId>\n            <requesterVpcInfo>\n               <ownerId>777788889999</ownerId>\n                <vpcId>vpc-1a2b3c4d</vpcId>\n              <cidrBlock>172.31.0.0/16</cidrBlock>\n           </requesterVpcInfo>\n           <accepterVpcInfo>\n                <ownerId>111122223333</ownerId>\n               <vpcId>vpc-aa22cc33</vpcId>\n           </accepterVpcInfo>\n            <status>\n                <code>pending-acceptance</code>\n               <message>Pending Acceptance by 111122223333</message>\n            </status>\n           <expirationTime>2014-02-17T16:00:50.000Z</expirationTime>\n        </item>\n        <item>\n           <vpcPeeringConnectionId>pcx-444bbb88</vpcPeeringConnectionId>\n            <requesterVpcInfo>\n               <ownerId>1237897234</ownerId>\n                <vpcId>vpc-2398abcd</vpcId>\n              <cidrBlock>172.30.0.0/16</cidrBlock>\n           </requesterVpcInfo>\n           <accepterVpcInfo>\n                <ownerId>98654313</ownerId>\n               <vpcId>vpc-0983bcda</vpcId>\n           </accepterVpcInfo>\n            <status>\n                <code>pending-acceptance</code>\n               <message>Pending Acceptance by 98654313</message>\n            </status>\n           <expirationTime>2015-02-17T16:00:50.000Z</expirationTime>\n        </item>\n    </vpcPeeringConnectionSet>\n</DescribeVpcPeeringConnectionsResponse>'
    connection_class = VPCConnection

    def default_body(self):
        return self.DESCRIBE_VPC_PEERING_CONNECTIONS

    def test_get_vpc_peering_connections(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_vpc_peering_connections(['pcx-111aaa22', 'pcx-444bbb88'], filters=[('status-code', ['pending-acceptance'])])
        self.assertEqual(len(api_response), 2)
        for vpc_peering_connection in api_response:
            if vpc_peering_connection.id == 'pcx-111aaa22':
                self.assertEqual(vpc_peering_connection.id, 'pcx-111aaa22')
                self.assertEqual(vpc_peering_connection.status_code, 'pending-acceptance')
                self.assertEqual(vpc_peering_connection.status_message, 'Pending Acceptance by 111122223333')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '777788889999')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-1a2b3c4d')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '172.31.0.0/16')
                self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '111122223333')
                self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-aa22cc33')
                self.assertEqual(vpc_peering_connection.expiration_time, '2014-02-17T16:00:50.000Z')
            else:
                self.assertEqual(vpc_peering_connection.id, 'pcx-444bbb88')
                self.assertEqual(vpc_peering_connection.status_code, 'pending-acceptance')
                self.assertEqual(vpc_peering_connection.status_message, 'Pending Acceptance by 98654313')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '1237897234')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-2398abcd')
                self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '172.30.0.0/16')
                self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '98654313')
                self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-0983bcda')
                self.assertEqual(vpc_peering_connection.expiration_time, '2015-02-17T16:00:50.000Z')
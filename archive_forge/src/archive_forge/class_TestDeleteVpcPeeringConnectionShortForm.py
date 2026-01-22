from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
class TestDeleteVpcPeeringConnectionShortForm(unittest.TestCase):
    DESCRIBE_VPC_PEERING_CONNECTIONS = b'<?xml version="1.0" encoding="UTF-8"?>\n<DescribeVpcPeeringConnectionsResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n   <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n   <vpcPeeringConnectionSet>\n      <item>\n         <vpcPeeringConnectionId>pcx-111aaa22</vpcPeeringConnectionId>\n         <requesterVpcInfo>\n            <ownerId>777788889999</ownerId>\n            <vpcId>vpc-1a2b3c4d</vpcId>\n            <cidrBlock>172.31.0.0/16</cidrBlock>\n         </requesterVpcInfo>\n         <accepterVpcInfo>\n            <ownerId>111122223333</ownerId>\n            <vpcId>vpc-aa22cc33</vpcId>\n         </accepterVpcInfo>\n         <status>\n            <code>pending-acceptance</code>\n            <message>Pending Acceptance by 111122223333</message>\n         </status>\n         <expirationTime>2014-02-17T16:00:50.000Z</expirationTime>\n      </item>\n   </vpcPeeringConnectionSet>\n</DescribeVpcPeeringConnectionsResponse>'
    DELETE_VPC_PEERING_CONNECTION = b'<DeleteVpcPeeringConnectionResponse xmlns="http://ec2.amazonaws.com/doc/2014-05-01/">\n  <requestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</requestId>\n  <return>true</return>\n</DeleteVpcPeeringConnectionResponse>'

    def test_delete_vpc_peering_connection(self):
        vpc_conn = VPCConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        mock_response = mock.Mock()
        mock_response.read.return_value = self.DESCRIBE_VPC_PEERING_CONNECTIONS
        mock_response.status = 200
        vpc_conn.make_request = mock.Mock(return_value=mock_response)
        vpc_peering_connections = vpc_conn.get_all_vpc_peering_connections()
        self.assertEquals(1, len(vpc_peering_connections))
        vpc_peering_connection = vpc_peering_connections[0]
        mock_response = mock.Mock()
        mock_response.read.return_value = self.DELETE_VPC_PEERING_CONNECTION
        mock_response.status = 200
        vpc_conn.make_request = mock.Mock(return_value=mock_response)
        self.assertEquals(True, vpc_peering_connection.delete())
        self.assertIn('DeleteVpcPeeringConnection', vpc_conn.make_request.call_args_list[0][0])
        self.assertNotIn('DeleteVpc', vpc_conn.make_request.call_args_list[0][0])
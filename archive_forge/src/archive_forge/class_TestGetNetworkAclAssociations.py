from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
class TestGetNetworkAclAssociations(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return b'\n        <DescribeNetworkAclsResponse xmlns="http://ec2.amazonaws.com/doc/2013-10-01/">\n            <requestId>59dbff89-35bd-4eac-99ed-be587EXAMPLE</requestId>\n            <networkAclSet>\n            <item>\n              <networkAclId>acl-5d659634</networkAclId>\n              <vpcId>vpc-5266953b</vpcId>\n              <default>false</default>\n              <entrySet>\n                <item>\n                  <ruleNumber>110</ruleNumber>\n                  <protocol>6</protocol>\n                  <ruleAction>allow</ruleAction>\n                  <egress>true</egress>\n                  <cidrBlock>0.0.0.0/0</cidrBlock>\n                  <portRange>\n                    <from>49152</from>\n                    <to>65535</to>\n                  </portRange>\n                </item>\n              </entrySet>\n              <associationSet>\n                <item>\n                  <networkAclAssociationId>aclassoc-c26596ab</networkAclAssociationId>\n                  <networkAclId>acl-5d659634</networkAclId>\n                  <subnetId>subnet-f0669599</subnetId>\n                </item>\n              </associationSet>\n              <tagSet/>\n            </item>\n          </networkAclSet>\n        </DescribeNetworkAclsResponse>\n    '

    def test_get_network_acl_associations(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_network_acls()
        association = api_response[0].associations[0]
        self.assertEqual(association.network_acl_id, 'acl-5d659634')
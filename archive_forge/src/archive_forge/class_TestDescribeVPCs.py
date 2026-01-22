from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestDescribeVPCs(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def default_body(self):
        return DESCRIBE_VPCS

    def test_get_vpcs(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.get_all_vpcs()
        self.assertEqual(len(api_response), 1)
        vpc = api_response[0]
        self.assertFalse(vpc.is_default)
        self.assertEqual(vpc.instance_tenancy, 'default')
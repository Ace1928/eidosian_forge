from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_vpc_update_classic_link_enabled(self):
    self.vpc.classic_link_enabled = False
    self.set_http_response(status_code=200)
    self.vpc.update_classic_link_enabled(dry_run=True, validate=True)
    self.assert_request_parameters({'Action': 'DescribeVpcClassicLink', 'VpcId.1': self.vpc_id, 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(self.vpc.classic_link_enabled, 'true')
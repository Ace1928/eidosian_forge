from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestEnableClassicLinkVpc(TestVpcClassicLink):

    def default_body(self):
        return b'\n            <EnableVpcClassicLinkResponse xmlns="http://ec2.amazonaws.com/doc/2014-09-01/"> \n                <requestId>4ab2b2b3-a267-4366-a070-bab853b5927d</requestId>\n                <return>true</return>\n            </EnableVpcClassicLinkResponse>\n        '

    def test_enable_classic_link(self):
        self.set_http_response(status_code=200)
        response = self.vpc.enable_classic_link(dry_run=True)
        self.assertTrue(response)
        self.assert_request_parameters({'Action': 'EnableVpcClassicLink', 'VpcId': self.vpc_id, 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
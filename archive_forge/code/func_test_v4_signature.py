import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_v4_signature(self):
    params = {'Action': 'DescribeInstances', 'Version': '2013-10-15'}
    headers = {'Host': 'ec2.eu-west-1.amazonaws.com', 'Accept-Encoding': 'gzip,deflate', 'X-AMZ-Date': '20150304T173452Z', 'User-Agent': 'libcloud/0.17.0 (Amazon EC2 (eu-central-1)) '}
    dt = self.now
    sig = self.signer._get_authorization_v4_header(params=params, headers=headers, dt=dt, method='GET', path='/my_action/')
    self.assertEqual(sig, 'AWS4-HMAC-SHA256 Credential=my_key/20150304/my_region/my_service/aws4_request, SignedHeaders=accept-encoding;host;user-agent;x-amz-date, Signature=f9868f8414b3c3f856c7955019cc1691265541f5162b9b772d26044280d39bd3')
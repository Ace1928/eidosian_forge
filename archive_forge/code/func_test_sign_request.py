import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common import aliyun
from libcloud.common.aliyun import AliyunRequestSignerAlgorithmV1_0
def test_sign_request(self):
    params = {'TimeStamp': '2012-12-26T10:33:56Z', 'Format': 'XML', 'AccessKeyId': 'testid', 'Action': 'DescribeRegions', 'SignatureMethod': 'HMAC-SHA1', 'RegionId': 'region1', 'SignatureNonce': 'NwDAxvLU6tFE0DVb', 'Version': '2014-05-26', 'SignatureVersion': '1.0'}
    method = 'GET'
    path = '/'
    expected = 'K9fCVP6Jrklpd3rLYKh1pfrrFNo='
    self.assertEqual(expected, self.signer._sign_request(params, method, path))
import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_v4_signature_contains_signed_headers(self):
    with mock.patch('libcloud.common.aws.AWSRequestSignerAlgorithmV4._get_signed_headers') as mock_get_headers:
        mock_get_headers.return_value = 'my_signed_headers'
        sig = self.signer._get_authorization_v4_header({}, {}, self.now, method='GET', path='/')
    self.assertIn('SignedHeaders=my_signed_headers, ', sig)
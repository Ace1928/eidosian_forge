import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_v4_signature_contains_signature(self):
    with mock.patch('libcloud.common.aws.AWSRequestSignerAlgorithmV4._get_signature') as mock_get_signature:
        mock_get_signature.return_value = 'my_signature'
        sig = self.signer._get_authorization_v4_header({}, {}, self.now)
    self.assertIn('Signature=my_signature', sig)
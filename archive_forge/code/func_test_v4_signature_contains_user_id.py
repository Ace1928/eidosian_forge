import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_v4_signature_contains_user_id(self):
    sig = self.signer._get_authorization_v4_header(params={}, headers={}, dt=self.now)
    self.assertIn('Credential=my_key/', sig)
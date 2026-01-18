from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_passed_in_values_are_used(self):
    p = provider.Provider('aws', 'access_key', 'secret_key', 'security_token')
    self.assertEqual(p.access_key, 'access_key')
    self.assertEqual(p.secret_key, 'secret_key')
    self.assertEqual(p.security_token, 'security_token')
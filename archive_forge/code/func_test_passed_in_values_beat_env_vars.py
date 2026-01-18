from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_passed_in_values_beat_env_vars(self):
    self.environ['AWS_ACCESS_KEY_ID'] = 'env_access_key'
    self.environ['AWS_SECRET_ACCESS_KEY'] = 'env_secret_key'
    self.environ['AWS_SECURITY_TOKEN'] = 'env_security_token'
    p = provider.Provider('aws', 'access_key', 'secret_key')
    self.assertEqual(p.access_key, 'access_key')
    self.assertEqual(p.secret_key, 'secret_key')
    self.assertEqual(p.security_token, None)
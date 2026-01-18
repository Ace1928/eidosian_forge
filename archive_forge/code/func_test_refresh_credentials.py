from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_refresh_credentials(self):
    now = datetime.utcnow()
    first_expiration = (now + timedelta(seconds=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
    credentials = {u'AccessKeyId': u'first_access_key', u'Code': u'Success', u'Expiration': first_expiration, u'LastUpdated': u'2012-08-31T21:43:40Z', u'SecretAccessKey': u'first_secret_key', u'Token': u'first_token', u'Type': u'AWS-HMAC'}
    instance_config = {'allowall': credentials}
    self.get_instance_metadata.return_value = instance_config
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'first_access_key')
    self.assertEqual(p.secret_key, 'first_secret_key')
    self.assertEqual(p.security_token, 'first_token')
    self.assertIsNotNone(p._credential_expiry_time)
    expired = now - timedelta(seconds=20)
    p._credential_expiry_time = expired
    credentials['AccessKeyId'] = 'second_access_key'
    credentials['SecretAccessKey'] = 'second_secret_key'
    credentials['Token'] = 'second_token'
    self.get_instance_metadata.return_value = instance_config
    self.assertEqual(p.access_key, 'second_access_key')
    self.assertEqual(p.secret_key, 'second_secret_key')
    self.assertEqual(p.security_token, 'second_token')
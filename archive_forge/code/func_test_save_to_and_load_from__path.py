import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_save_to_and_load_from__path(self):
    temp_dir = self.useFixture(TempDir()).path
    credentials_path = os.path.join(temp_dir, 'credentials')
    credentials = OAuthAuthorizer('consumer.key', consumer_secret='consumer.secret', access_token=AccessToken('access.key', 'access.secret'))
    credentials.save_to_path(credentials_path)
    self.assertTrue(os.path.exists(credentials_path))
    self.assertEqual(stat.S_IMODE(os.stat(credentials_path).st_mode), stat.S_IREAD | stat.S_IWRITE)
    loaded_credentials = OAuthAuthorizer.load_from_path(credentials_path)
    self.assertEqual(loaded_credentials.consumer.key, 'consumer.key')
    self.assertEqual(loaded_credentials.consumer.secret, 'consumer.secret')
    self.assertEqual(loaded_credentials.access_token.key, 'access.key')
    self.assertEqual(loaded_credentials.access_token.secret, 'access.secret')
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tsigkey_name
from designateclient.functionaltests.datagen import random_tsigkey_secret
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import TSIGKeyFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_tsigkey_set(self):
    client = self.clients.as_user('admin')
    updated_name = random_tsigkey_name('updated')
    tsigkey = client.tsigkey_set(self.tsigkey.id, name=updated_name, secret='An updated tsigsecret')
    self.assertEqual(tsigkey.secret, 'An updated tsigsecret')
    self.assertEqual(tsigkey.name, updated_name)
from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
def test_tld_set(self):
    client = self.clients.as_user('admin')
    updated_name = random_tld('updated')
    tld = client.tld_set(self.tld.id, name=updated_name, description='An updated tld')
    self.assertEqual(tld.description, 'An updated tld')
    self.assertEqual(tld.name, updated_name)
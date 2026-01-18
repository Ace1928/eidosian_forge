from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_zone_blacklist_set(self):
    client = self.clients.as_user('admin')
    updated_pattern = random_blacklist('updatedblacklist')
    blacklist = client.zone_blacklist_set(id=self.blacklist.id, pattern=updated_pattern, description='An updated blacklist')
    self.assertEqual(blacklist.created_at, self.blacklist.created_at)
    self.assertEqual(blacklist.description, 'An updated blacklist')
    self.assertEqual(blacklist.id, self.blacklist.id)
    self.assertEqual(blacklist.pattern, updated_pattern)
    self.assertNotEqual(blacklist.updated_at, self.blacklist.updated_at)
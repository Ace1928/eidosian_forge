from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_zone_blacklist_set_no_description(self):
    client = self.clients.as_user('admin')
    blacklist = client.zone_blacklist_set(id=self.blacklist.id, no_description=True)
    self.assertEqual(blacklist.description, 'None')
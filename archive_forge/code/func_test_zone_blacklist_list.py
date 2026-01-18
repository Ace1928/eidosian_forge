from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
def test_zone_blacklist_list(self):
    blacklists = self.clients.as_user('admin').zone_blacklist_list()
    self.assertGreater(len(blacklists), 0)
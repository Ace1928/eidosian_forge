from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_set(self):
    ttl = int(self.zone.ttl) + 123
    email = f'updated{self.zone.email}'
    description = 'new description'
    zone = self.clients.zone_set(self.zone.id, ttl=ttl, email=email, description=description)
    self.assertEqual(ttl, int(zone.ttl))
    self.assertEqual(email, zone.email)
    self.assertEqual(description, zone.description)
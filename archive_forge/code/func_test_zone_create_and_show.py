from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_zone_create_and_show(self):
    zone = self.clients.zone_show(self.zone.id)
    self.assertTrue(hasattr(zone, 'action'))
    self.assertEqual(self.zone.created_at, zone.created_at)
    self.assertEqual(self.zone.description, zone.description)
    self.assertEqual(self.zone.email, zone.email)
    self.assertEqual(self.zone.id, zone.id)
    self.assertEqual(self.zone.masters, zone.masters)
    self.assertEqual(self.zone.name, zone.name)
    self.assertEqual(self.zone.pool_id, zone.pool_id)
    self.assertEqual(self.zone.project_id, zone.project_id)
    self.assertEqual(self.zone.serial, zone.serial)
    self.assertTrue(hasattr(zone, 'status'))
    self.assertEqual(self.zone.transferred_at, zone.transferred_at)
    self.assertEqual(self.zone.ttl, zone.ttl)
    self.assertEqual(self.zone.type, zone.type)
    self.assertEqual(self.zone.updated_at, zone.updated_at)
    self.assertEqual(self.zone.version, zone.version)
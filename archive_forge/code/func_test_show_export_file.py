from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import ExportFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_show_export_file(self):
    zone_export = self.useFixture(ExportFixture(zone=self.zone)).zone_export
    fetched_export = self.clients.zone_export_showfile(zone_export.id)
    self.assertIn('$ORIGIN', fetched_export.data)
    self.assertIn('$TTL', fetched_export.data)
    self.assertIn('SOA', fetched_export.data)
    self.assertIn('NS', fetched_export.data)
    self.assertIn(self.zone.name, fetched_export.data)
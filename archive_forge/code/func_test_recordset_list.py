from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_a_recordset_name
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import RecordsetFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_recordset_list(self):
    rsets = self.clients.recordset_list(self.zone.id)
    self.assertGreater(len(rsets), 0)
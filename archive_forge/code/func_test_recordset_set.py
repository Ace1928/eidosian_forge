from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_a_recordset_name
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import RecordsetFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
def test_recordset_set(self):
    rset = self.clients.recordset_set(self.zone.id, self.recordset.id, record='2.3.4.5', ttl=2345, description='Updated description')
    self.assertEqual(rset.records, '2.3.4.5')
    self.assertEqual(rset.ttl, '2345')
    self.assertEqual(rset.description, 'Updated description')
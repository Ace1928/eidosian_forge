from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import base
class TestZoneTransferAccept(base.TestCase):

    def test_basic(self):
        sot = zone_transfer.ZoneTransferAccept()
        self.assertEqual('transfer_accepts', sot.resources_key)
        self.assertEqual('/zones/tasks/transfer_accepts', sot.base_path)
        self.assertTrue(sot.allow_list)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'status': 'status'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = zone_transfer.ZoneTransferAccept(**EXAMPLE_ACCEPT)
        self.assertEqual(IDENTIFIER, sot.id)
        self.assertEqual(EXAMPLE_ACCEPT['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE_ACCEPT['updated_at'], sot.updated_at)
        self.assertEqual(EXAMPLE_ACCEPT['key'], sot.key)
        self.assertEqual(EXAMPLE_ACCEPT['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE_ACCEPT['status'], sot.status)
        self.assertEqual(EXAMPLE_ACCEPT['zone_id'], sot.zone_id)
        self.assertEqual(EXAMPLE_ACCEPT['zone_transfer_request_id'], sot.zone_transfer_request_id)
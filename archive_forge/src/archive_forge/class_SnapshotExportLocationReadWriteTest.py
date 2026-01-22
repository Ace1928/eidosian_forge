import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.ddt
@testtools.skipUnless(CONF.run_snapshot_tests and CONF.run_mount_snapshot_tests, 'Snapshots or mountable snapshots tests are disabled.')
@utils.skip_if_microversion_not_supported('2.32')
class SnapshotExportLocationReadWriteTest(base.BaseTestCase):

    def setUp(self):
        super(SnapshotExportLocationReadWriteTest, self).setUp()
        self.share = self.create_share(client=self.get_user_client())
        self.snapshot = self.create_snapshot(share=self.share['id'], client=self.get_user_client())

    @ddt.data('admin', 'user')
    def test_get_snapshot_export_location(self, role):
        client = self.admin_client if role == 'admin' else self.user_client
        export_locations = client.list_snapshot_export_locations(self.snapshot['id'])
        el = client.get_snapshot_export_location(self.snapshot['id'], export_locations[0]['ID'])
        expected_keys = ['path', 'id', 'updated_at', 'created_at']
        if role == 'admin':
            expected_keys.extend(['is_admin_only', 'share_snapshot_instance_id'])
            self.assertTrue(uuidutils.is_uuid_like(el['share_snapshot_instance_id']))
            self.assertIn(el['is_admin_only'], ('True', 'False'))
        self.assertTrue(uuidutils.is_uuid_like(el['id']))
        for key in expected_keys:
            self.assertIn(key, el)

    @ddt.data('admin', 'user')
    def test_list_snapshot_export_locations(self, role):
        client = self.admin_client if role == 'admin' else self.user_client
        export_locations = client.list_snapshot_export_locations(self.snapshot['id'])
        self.assertGreater(len(export_locations), 0)
        expected_keys = ('ID', 'Path')
        for el in export_locations:
            for key in expected_keys:
                self.assertIn(key, el)
            self.assertTrue(uuidutils.is_uuid_like(el['ID']))

    @ddt.data('admin', 'user')
    def test_list_snapshot_export_locations_with_columns(self, role):
        client = self.admin_client if role == 'admin' else self.user_client
        export_locations = client.list_snapshot_export_locations(self.snapshot['id'], columns='id,path')
        self.assertGreater(len(export_locations), 0)
        expected_keys = ('Id', 'Path')
        unexpected_keys = ('Updated At', 'Created At')
        for el in export_locations:
            for key in expected_keys:
                self.assertIn(key, el)
            for key in unexpected_keys:
                self.assertNotIn(key, el)
            self.assertTrue(uuidutils.is_uuid_like(el['Id']))
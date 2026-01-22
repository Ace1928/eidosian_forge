from unittest import mock
from openstack.compute.v2 import server_migration
from openstack.tests.unit import base
class TestServerMigration(base.TestCase):

    def setUp(self):
        super().setUp()
        self.resp = mock.Mock()
        self.resp.body = None
        self.resp.json = mock.Mock(return_value=self.resp.body)
        self.resp.status_code = 200
        self.sess = mock.Mock()
        self.sess.post = mock.Mock(return_value=self.resp)

    def test_basic(self):
        sot = server_migration.ServerMigration()
        self.assertEqual('migration', sot.resource_key)
        self.assertEqual('migrations', sot.resources_key)
        self.assertEqual('/servers/%(server_id)s/migrations', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_list)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)

    def test_make_it(self):
        sot = server_migration.ServerMigration(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['user_id'], sot.user_id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['source_compute'], sot.source_compute)
        self.assertEqual(EXAMPLE['source_node'], sot.source_node)
        self.assertEqual(EXAMPLE['dest_host'], sot.dest_host)
        self.assertEqual(EXAMPLE['dest_compute'], sot.dest_compute)
        self.assertEqual(EXAMPLE['dest_node'], sot.dest_node)
        self.assertEqual(EXAMPLE['memory_processed_bytes'], sot.memory_processed_bytes)
        self.assertEqual(EXAMPLE['memory_remaining_bytes'], sot.memory_remaining_bytes)
        self.assertEqual(EXAMPLE['memory_total_bytes'], sot.memory_total_bytes)
        self.assertEqual(EXAMPLE['disk_processed_bytes'], sot.disk_processed_bytes)
        self.assertEqual(EXAMPLE['disk_remaining_bytes'], sot.disk_remaining_bytes)
        self.assertEqual(EXAMPLE['disk_total_bytes'], sot.disk_total_bytes)

    @mock.patch.object(server_migration.ServerMigration, '_get_session', lambda self, x: x)
    def test_force_complete(self):
        sot = server_migration.ServerMigration(**EXAMPLE)
        self.assertIsNone(sot.force_complete(self.sess))
        url = 'servers/%s/migrations/%s/action' % (EXAMPLE['server_id'], EXAMPLE['id'])
        body = {'force_complete': None}
        self.sess.post.assert_called_with(url, microversion=mock.ANY, json=body)
from openstack.compute.v2 import migration
from openstack.tests.unit import base
class TestMigration(base.TestCase):

    def test_basic(self):
        sot = migration.Migration()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('migrations', sot.resources_key)
        self.assertEqual('/os-migrations', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'host': 'host', 'status': 'status', 'migration_type': 'migration_type', 'source_compute': 'source_compute', 'user_id': 'user_id', 'project_id': 'project_id', 'changes_since': 'changes-since', 'changes_before': 'changes-before', 'server_id': 'instance_uuid'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = migration.Migration(**EXAMPLE)
        self.assertEqual(EXAMPLE['uuid'], sot.id)
        self.assertEqual(EXAMPLE['instance_uuid'], sot.server_id)
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
        self.assertEqual(EXAMPLE['migration_type'], sot.migration_type)
        self.assertEqual(EXAMPLE['old_instance_type_id'], sot.old_flavor_id)
        self.assertEqual(EXAMPLE['new_instance_type_id'], sot.new_flavor_id)
from openstack.shared_file_system.v2 import share_group
from openstack.tests.unit import base
class TestShareGroups(base.TestCase):

    def test_basic(self):
        share_groups = share_group.ShareGroup()
        self.assertEqual('share_groups', share_groups.resources_key)
        self.assertEqual('/share-groups', share_groups.base_path)
        self.assertTrue(share_groups.allow_list)
        self.assertTrue(share_groups.allow_fetch)
        self.assertTrue(share_groups.allow_create)
        self.assertTrue(share_groups.allow_commit)
        self.assertTrue(share_groups.allow_delete)
        self.assertFalse(share_groups.allow_head)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'share_group_id': 'share_group_id'}, share_groups._query_mapping._mapping)

    def test_make_share_groups(self):
        share_group_res = share_group.ShareGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], share_group_res.id)
        self.assertEqual(EXAMPLE['status'], share_group_res.status)
        self.assertEqual(EXAMPLE['availability_zone'], share_group_res.availability_zone)
        self.assertEqual(EXAMPLE['description'], share_group_res.description)
        self.assertEqual(EXAMPLE['source_share_group_snapshot_id'], share_group_res.share_group_snapshot_id)
        self.assertEqual(EXAMPLE['share_network_id'], share_group_res.share_network_id)
        self.assertEqual(EXAMPLE['share_group_type_id'], share_group_res.share_group_type_id)
        self.assertEqual(EXAMPLE['consistent_snapshot_support'], share_group_res.consistent_snapshot_support)
        self.assertEqual(EXAMPLE['created_at'], share_group_res.created_at)
        self.assertEqual(EXAMPLE['project_id'], share_group_res.project_id)
        self.assertEqual(EXAMPLE['share_types'], share_group_res.share_types)
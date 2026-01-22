from openstack.shared_file_system.v2 import share_access_rule
from openstack.tests.unit import base
class TestShareAccessRule(base.TestCase):

    def test_basic(self):
        rules_resource = share_access_rule.ShareAccessRule()
        self.assertEqual('access_list', rules_resource.resources_key)
        self.assertEqual('/share-access-rules', rules_resource.base_path)
        self.assertTrue(rules_resource.allow_list)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'share_id': 'share_id'}, rules_resource._query_mapping._mapping)

    def test_make_share_access_rules(self):
        rules_resource = share_access_rule.ShareAccessRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], rules_resource.id)
        self.assertEqual(EXAMPLE['access_level'], rules_resource.access_level)
        self.assertEqual(EXAMPLE['state'], rules_resource.state)
        self.assertEqual(EXAMPLE['id'], rules_resource.id)
        self.assertEqual(EXAMPLE['access_type'], rules_resource.access_type)
        self.assertEqual(EXAMPLE['access_to'], rules_resource.access_to)
        self.assertEqual(EXAMPLE['access_key'], rules_resource.access_key)
        self.assertEqual(EXAMPLE['created_at'], rules_resource.created_at)
        self.assertEqual(EXAMPLE['updated_at'], rules_resource.updated_at)
        self.assertEqual(EXAMPLE['metadata'], rules_resource.metadata)
from openstack.identity.v3 import region
from openstack.tests.unit import base
class TestRegion(base.TestCase):

    def test_basic(self):
        sot = region.Region()
        self.assertEqual('region', sot.resource_key)
        self.assertEqual('regions', sot.resources_key)
        self.assertEqual('/regions', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('PATCH', sot.commit_method)
        self.assertDictEqual({'parent_region_id': 'parent_region_id', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = region.Region(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['links'], sot.links)
        self.assertEqual(EXAMPLE['parent_region_id'], sot.parent_region_id)
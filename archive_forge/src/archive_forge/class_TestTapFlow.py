from openstack.network.v2 import tap_flow
from openstack.tests.unit import base
class TestTapFlow(base.TestCase):

    def test_basic(self):
        sot = tap_flow.TapFlow()
        self.assertEqual('tap_flow', sot.resource_key)
        self.assertEqual('tap_flows', sot.resources_key)
        self.assertEqual('/taas/tap_flows', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = tap_flow.TapFlow(**EXAMPLE)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['source_port'], sot.source_port)
        self.assertEqual(EXAMPLE['tap_service_id'], sot.tap_service_id)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'name': 'name', 'project_id': 'project_id', 'sort_key': 'sort_key', 'sort_dir': 'sort_dir'}, sot._query_mapping._mapping)
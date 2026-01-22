from openstack.network.v2 import sfc_service_graph
from openstack.tests.unit import base
class TestSfcServiceGraph(base.TestCase):

    def test_basic(self):
        sot = sfc_service_graph.SfcServiceGraph()
        self.assertEqual('service_graph', sot.resource_key)
        self.assertEqual('service_graphs', sot.resources_key)
        self.assertEqual('/sfc/service_graphs', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = sfc_service_graph.SfcServiceGraph(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['port_chains'], sot.port_chains)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'name': 'name', 'project_id': 'project_id', 'tenant_id': 'tenant_id'}, sot._query_mapping._mapping)
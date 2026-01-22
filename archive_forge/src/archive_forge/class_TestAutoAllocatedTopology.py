from openstack.network.v2 import auto_allocated_topology
from openstack.tests.unit import base
class TestAutoAllocatedTopology(base.TestCase):

    def test_basic(self):
        topo = auto_allocated_topology.AutoAllocatedTopology
        self.assertEqual('auto_allocated_topology', topo.resource_key)
        self.assertEqual('/auto-allocated-topology', topo.base_path)
        self.assertFalse(topo.allow_create)
        self.assertTrue(topo.allow_fetch)
        self.assertFalse(topo.allow_commit)
        self.assertTrue(topo.allow_delete)
        self.assertFalse(topo.allow_list)

    def test_make_it(self):
        topo = auto_allocated_topology.AutoAllocatedTopology(**EXAMPLE)
        self.assertEqual(EXAMPLE['project_id'], topo.project_id)
from openstack.network.v2 import listener
from openstack.tests.unit import base
class TestListener(base.TestCase):

    def test_basic(self):
        sot = listener.Listener()
        self.assertEqual('listener', sot.resource_key)
        self.assertEqual('listeners', sot.resources_key)
        self.assertEqual('/lbaas/listeners', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = listener.Listener(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['connection_limit'], sot.connection_limit)
        self.assertEqual(EXAMPLE['default_pool_id'], sot.default_pool_id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['loadbalancers'], sot.load_balancer_ids)
        self.assertEqual(EXAMPLE['loadbalancer_id'], sot.load_balancer_id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['protocol_port'], sot.protocol_port)
        self.assertEqual(EXAMPLE['default_tls_container_ref'], sot.default_tls_container_ref)
        self.assertEqual(EXAMPLE['sni_container_refs'], sot.sni_container_refs)
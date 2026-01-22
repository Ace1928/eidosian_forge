from openstack.network.v2 import ndp_proxy
from openstack.tests.unit import base
class TestNDPProxy(base.TestCase):

    def test_basic(self):
        sot = ndp_proxy.NDPProxy()
        self.assertEqual('ndp_proxy', sot.resource_key)
        self.assertEqual('ndp_proxies', sot.resources_key)
        self.assertEqual('/ndp_proxies', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = ndp_proxy.NDPProxy(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['router_id'], sot.router_id)
        self.assertEqual(EXAMPLE['port_id'], sot.port_id)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['updated_at'], sot.updated_at)
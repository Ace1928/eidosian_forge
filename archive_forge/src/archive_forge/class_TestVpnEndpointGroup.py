from openstack.network.v2 import vpn_endpoint_group
from openstack.tests.unit import base
class TestVpnEndpointGroup(base.TestCase):

    def test_basic(self):
        sot = vpn_endpoint_group.VpnEndpointGroup()
        self.assertEqual('endpoint_group', sot.resource_key)
        self.assertEqual('endpoint_groups', sot.resources_key)
        self.assertEqual('/vpn/endpoint-groups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = vpn_endpoint_group.VpnEndpointGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['endpoints'], sot.endpoints)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'name': 'name', 'project_id': 'project_id', 'tenant_id': 'tenant_id', 'type': 'endpoint_type'}, sot._query_mapping._mapping)
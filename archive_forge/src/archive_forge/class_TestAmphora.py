import uuid
from openstack.load_balancer.v2 import amphora
from openstack.tests.unit import base
class TestAmphora(base.TestCase):

    def test_basic(self):
        test_amphora = amphora.Amphora()
        self.assertEqual('amphora', test_amphora.resource_key)
        self.assertEqual('amphorae', test_amphora.resources_key)
        self.assertEqual('/octavia/amphorae', test_amphora.base_path)
        self.assertFalse(test_amphora.allow_create)
        self.assertTrue(test_amphora.allow_fetch)
        self.assertFalse(test_amphora.allow_commit)
        self.assertFalse(test_amphora.allow_delete)
        self.assertTrue(test_amphora.allow_list)

    def test_make_it(self):
        test_amphora = amphora.Amphora(**EXAMPLE)
        self.assertEqual(IDENTIFIER, test_amphora.id)
        self.assertEqual(LB_ID, test_amphora.loadbalancer_id)
        self.assertEqual(COMPUTE_ID, test_amphora.compute_id)
        self.assertEqual(EXAMPLE['lb_network_ip'], test_amphora.lb_network_ip)
        self.assertEqual(EXAMPLE['vrrp_ip'], test_amphora.vrrp_ip)
        self.assertEqual(EXAMPLE['ha_ip'], test_amphora.ha_ip)
        self.assertEqual(VRRP_PORT_ID, test_amphora.vrrp_port_id)
        self.assertEqual(HA_PORT_ID, test_amphora.ha_port_id)
        self.assertEqual(EXAMPLE['cert_expiration'], test_amphora.cert_expiration)
        self.assertEqual(EXAMPLE['cert_busy'], test_amphora.cert_busy)
        self.assertEqual(EXAMPLE['role'], test_amphora.role)
        self.assertEqual(EXAMPLE['status'], test_amphora.status)
        self.assertEqual(EXAMPLE['vrrp_interface'], test_amphora.vrrp_interface)
        self.assertEqual(EXAMPLE['vrrp_id'], test_amphora.vrrp_id)
        self.assertEqual(EXAMPLE['vrrp_priority'], test_amphora.vrrp_priority)
        self.assertEqual(EXAMPLE['cached_zone'], test_amphora.cached_zone)
        self.assertEqual(EXAMPLE['created_at'], test_amphora.created_at)
        self.assertEqual(EXAMPLE['updated_at'], test_amphora.updated_at)
        self.assertEqual(IMAGE_ID, test_amphora.image_id)
        self.assertEqual(COMPUTE_FLAVOR, test_amphora.compute_flavor)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'id': 'id', 'loadbalancer_id': 'loadbalancer_id', 'compute_id': 'compute_id', 'lb_network_ip': 'lb_network_ip', 'vrrp_ip': 'vrrp_ip', 'ha_ip': 'ha_ip', 'vrrp_port_id': 'vrrp_port_id', 'ha_port_id': 'ha_port_id', 'cert_expiration': 'cert_expiration', 'cert_busy': 'cert_busy', 'role': 'role', 'status': 'status', 'vrrp_interface': 'vrrp_interface', 'vrrp_id': 'vrrp_id', 'vrrp_priority': 'vrrp_priority', 'cached_zone': 'cached_zone', 'created_at': 'created_at', 'updated_at': 'updated_at', 'image_id': 'image_id', 'image_id': 'image_id'}, test_amphora._query_mapping._mapping)
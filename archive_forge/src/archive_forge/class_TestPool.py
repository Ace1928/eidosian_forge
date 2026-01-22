from openstack.network.v2 import pool
from openstack.tests.unit import base
class TestPool(base.TestCase):

    def test_basic(self):
        sot = pool.Pool()
        self.assertEqual('pool', sot.resource_key)
        self.assertEqual('pools', sot.resources_key)
        self.assertEqual('/lbaas/pools', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = pool.Pool(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['healthmonitor_id'], sot.health_monitor_id)
        self.assertEqual(EXAMPLE['health_monitors'], sot.health_monitor_ids)
        self.assertEqual(EXAMPLE['health_monitor_status'], sot.health_monitor_status)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['lb_algorithm'], sot.lb_algorithm)
        self.assertEqual(EXAMPLE['listeners'], sot.listener_ids)
        self.assertEqual(EXAMPLE['listener_id'], sot.listener_id)
        self.assertEqual(EXAMPLE['members'], sot.member_ids)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['provider'], sot.provider)
        self.assertEqual(EXAMPLE['session_persistence'], sot.session_persistence)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['status_description'], sot.status_description)
        self.assertEqual(EXAMPLE['subnet_id'], sot.subnet_id)
        self.assertEqual(EXAMPLE['loadbalancers'], sot.load_balancer_ids)
        self.assertEqual(EXAMPLE['loadbalancer_id'], sot.load_balancer_id)
        self.assertEqual(EXAMPLE['vip_id'], sot.virtual_ip_id)
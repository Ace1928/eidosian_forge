from unittest import mock
import uuid
from openstack.load_balancer.v2 import load_balancer
from openstack.tests.unit import base
class TestLoadBalancerFailover(base.TestCase):

    def test_basic(self):
        test_load_balancer = load_balancer.LoadBalancerFailover()
        self.assertEqual('/lbaas/loadbalancers/%(lb_id)s/failover', test_load_balancer.base_path)
        self.assertFalse(test_load_balancer.allow_create)
        self.assertFalse(test_load_balancer.allow_fetch)
        self.assertFalse(test_load_balancer.allow_delete)
        self.assertFalse(test_load_balancer.allow_list)
        self.assertTrue(test_load_balancer.allow_commit)
from openstack.network.v2 import quota
from openstack import resource
from openstack.tests.unit import base
class TestQuota(base.TestCase):

    def test_basic(self):
        sot = quota.Quota()
        self.assertEqual('quota', sot.resource_key)
        self.assertEqual('quotas', sot.resources_key)
        self.assertEqual('/quotas', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = quota.Quota(**EXAMPLE)
        self.assertEqual(EXAMPLE['floatingip'], sot.floating_ips)
        self.assertEqual(EXAMPLE['network'], sot.networks)
        self.assertEqual(EXAMPLE['port'], sot.ports)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertEqual(EXAMPLE['router'], sot.routers)
        self.assertEqual(EXAMPLE['subnet'], sot.subnets)
        self.assertEqual(EXAMPLE['subnetpool'], sot.subnet_pools)
        self.assertEqual(EXAMPLE['security_group_rule'], sot.security_group_rules)
        self.assertEqual(EXAMPLE['security_group'], sot.security_groups)
        self.assertEqual(EXAMPLE['rbac_policy'], sot.rbac_policies)
        self.assertEqual(EXAMPLE['healthmonitor'], sot.health_monitors)
        self.assertEqual(EXAMPLE['listener'], sot.listeners)
        self.assertEqual(EXAMPLE['loadbalancer'], sot.load_balancers)
        self.assertEqual(EXAMPLE['l7policy'], sot.l7_policies)
        self.assertEqual(EXAMPLE['pool'], sot.pools)
        self.assertEqual(EXAMPLE['check_limit'], sot.check_limit)

    def test_prepare_request(self):
        body = {'id': 'ABCDEFGH', 'network': '12345'}
        quota_obj = quota.Quota(**body)
        response = quota_obj._prepare_request()
        self.assertNotIn('id', response)

    def test_alternate_id(self):
        my_project_id = 'my-tenant-id'
        body = {'project_id': my_project_id, 'network': 12345}
        quota_obj = quota.Quota(**body)
        self.assertEqual(my_project_id, resource.Resource._get_id(quota_obj))
import testtools
from openstack.network.v2 import firewall_policy
class TestFirewallPolicy(testtools.TestCase):

    def test_basic(self):
        sot = firewall_policy.FirewallPolicy()
        self.assertEqual('firewall_policy', sot.resource_key)
        self.assertEqual('firewall_policies', sot.resources_key)
        self.assertEqual('/fwaas/firewall_policies', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = firewall_policy.FirewallPolicy(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['firewall_rules'], sot.firewall_rules)
        self.assertEqual(EXAMPLE['shared'], sot.shared)
        self.assertEqual(list, type(sot.firewall_rules))
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
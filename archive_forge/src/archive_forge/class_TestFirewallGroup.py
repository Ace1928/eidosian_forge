import testtools
from openstack.network.v2 import firewall_group
class TestFirewallGroup(testtools.TestCase):

    def test_basic(self):
        sot = firewall_group.FirewallGroup()
        self.assertEqual('firewall_group', sot.resource_key)
        self.assertEqual('firewall_groups', sot.resources_key)
        self.assertEqual('/fwaas/firewall_groups', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = firewall_group.FirewallGroup(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['egress_firewall_policy_id'], sot.egress_firewall_policy_id)
        self.assertEqual(EXAMPLE['ingress_firewall_policy_id'], sot.ingress_firewall_policy_id)
        self.assertEqual(EXAMPLE['shared'], sot.shared)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(list, type(sot.ports))
        self.assertEqual(EXAMPLE['ports'], sot.ports)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
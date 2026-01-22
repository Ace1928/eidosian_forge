import testtools
from openstack.network.v2 import firewall_rule
class TestFirewallRule(testtools.TestCase):

    def test_basic(self):
        sot = firewall_rule.FirewallRule()
        self.assertEqual('firewall_rule', sot.resource_key)
        self.assertEqual('firewall_rules', sot.resources_key)
        self.assertEqual('/fwaas/firewall_rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = firewall_rule.FirewallRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['action'], sot.action)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['destination_ip_address'], sot.destination_ip_address)
        self.assertEqual(EXAMPLE['destination_port'], sot.destination_port)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['enabled'], sot.enabled)
        self.assertEqual(EXAMPLE['ip_version'], sot.ip_version)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['shared'], sot.shared)
        self.assertEqual(EXAMPLE['source_ip_address'], sot.source_ip_address)
        self.assertEqual(EXAMPLE['source_port'], sot.source_port)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
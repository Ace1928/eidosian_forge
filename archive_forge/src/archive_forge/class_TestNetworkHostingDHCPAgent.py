from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
class TestNetworkHostingDHCPAgent(base.TestCase):

    def test_basic(self):
        net = agent.NetworkHostingDHCPAgent()
        self.assertEqual('agent', net.resource_key)
        self.assertEqual('agents', net.resources_key)
        self.assertEqual('/networks/%(network_id)s/dhcp-agents', net.base_path)
        self.assertEqual('dhcp-agent', net.resource_name)
        self.assertFalse(net.allow_create)
        self.assertTrue(net.allow_fetch)
        self.assertFalse(net.allow_commit)
        self.assertFalse(net.allow_delete)
        self.assertTrue(net.allow_list)
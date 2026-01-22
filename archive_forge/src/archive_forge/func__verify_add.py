from openstack.network.v2 import network
from openstack.tests.functional import base
def _verify_add(self, network):
    net = self.user_cloud.network.dhcp_agent_hosting_networks(self.AGENT_ID)
    net_ids = [n.id for n in net]
    self.assertIn(self.NETWORK_ID, net_ids)
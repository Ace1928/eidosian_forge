from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
def test_remove_agent_from_network(self):
    net = agent.Agent(**EXAMPLE)
    sess = mock.Mock()
    network_id = {}
    self.assertIsNone(net.remove_agent_from_network(sess, network_id))
    body = {'network_id': {}}
    sess.delete.assert_called_with('agents/IDENTIFIER/dhcp-networks/', json=body)
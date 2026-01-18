from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
def test_remove_router_from_agent(self):
    sot = agent.Agent(**EXAMPLE)
    sess = mock.Mock()
    router_id = {}
    self.assertIsNone(sot.remove_router_from_agent(sess, router_id))
    body = {'router_id': {}}
    sess.delete.assert_called_with('agents/IDENTIFIER/l3-routers/', json=body)
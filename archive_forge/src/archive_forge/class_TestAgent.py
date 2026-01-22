from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
class TestAgent(base.TestCase):

    def test_basic(self):
        sot = agent.Agent()
        self.assertEqual('agent', sot.resource_key)
        self.assertEqual('agents', sot.resources_key)
        self.assertEqual('/agents', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = agent.Agent(**EXAMPLE)
        self.assertTrue(sot.is_admin_state_up)
        self.assertEqual(EXAMPLE['agent_type'], sot.agent_type)
        self.assertTrue(sot.is_alive)
        self.assertEqual(EXAMPLE['availability_zone'], sot.availability_zone)
        self.assertEqual(EXAMPLE['binary'], sot.binary)
        self.assertEqual(EXAMPLE['configurations'], sot.configuration)
        self.assertEqual(EXAMPLE['created_at'], sot.created_at)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['heartbeat_timestamp'], sot.last_heartbeat_at)
        self.assertEqual(EXAMPLE['host'], sot.host)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['resources_synced'], sot.resources_synced)
        self.assertEqual(EXAMPLE['started_at'], sot.started_at)
        self.assertEqual(EXAMPLE['topic'], sot.topic)
        self.assertEqual(EXAMPLE['ha_state'], sot.ha_state)

    def test_add_agent_to_network(self):
        net = agent.Agent(**EXAMPLE)
        response = mock.Mock()
        response.body = {'network_id': '1'}
        response.json = mock.Mock(return_value=response.body)
        sess = mock.Mock()
        sess.post = mock.Mock(return_value=response)
        body = {'network_id': '1'}
        self.assertEqual(response.body, net.add_agent_to_network(sess, **body))
        url = 'agents/IDENTIFIER/dhcp-networks'
        sess.post.assert_called_with(url, json=body)

    def test_remove_agent_from_network(self):
        net = agent.Agent(**EXAMPLE)
        sess = mock.Mock()
        network_id = {}
        self.assertIsNone(net.remove_agent_from_network(sess, network_id))
        body = {'network_id': {}}
        sess.delete.assert_called_with('agents/IDENTIFIER/dhcp-networks/', json=body)

    def test_add_router_to_agent(self):
        sot = agent.Agent(**EXAMPLE)
        response = mock.Mock()
        response.body = {'router_id': '1'}
        response.json = mock.Mock(return_value=response.body)
        sess = mock.Mock()
        sess.post = mock.Mock(return_value=response)
        router_id = '1'
        self.assertEqual(response.body, sot.add_router_to_agent(sess, router_id))
        body = {'router_id': router_id}
        url = 'agents/IDENTIFIER/l3-routers'
        sess.post.assert_called_with(url, json=body)

    def test_remove_router_from_agent(self):
        sot = agent.Agent(**EXAMPLE)
        sess = mock.Mock()
        router_id = {}
        self.assertIsNone(sot.remove_router_from_agent(sess, router_id))
        body = {'router_id': {}}
        sess.delete.assert_called_with('agents/IDENTIFIER/l3-routers/', json=body)

    def test_get_bgp_speakers_hosted_by_dragent(self):
        sot = agent.Agent(**EXAMPLE)
        sess = mock.Mock()
        response = mock.Mock()
        response.body = {'bgp_speakers': [{'name': 'bgp_speaker_1', 'ip_version': 4}]}
        response.json = mock.Mock(return_value=response.body)
        response.status_code = 200
        sess.get = mock.Mock(return_value=response)
        resp = sot.get_bgp_speakers_hosted_by_dragent(sess)
        self.assertEqual(resp, response.body)
        sess.get.assert_called_with('agents/IDENTIFIER/bgp-drinstances')
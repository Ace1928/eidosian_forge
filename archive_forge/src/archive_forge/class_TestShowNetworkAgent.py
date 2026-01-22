from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkAgent(TestNetworkAgent):
    _network_agent = network_fakes.create_one_network_agent()
    columns = ('admin_state_up', 'agent_type', 'alive', 'availability_zone', 'binary', 'configuration', 'created_at', 'description', 'host', 'ha_state', 'id', 'last_heartbeat_at', 'resources_synced', 'started_at', 'topic')
    data = (network_agent.AdminStateColumn(_network_agent.is_admin_state_up), _network_agent.agent_type, network_agent.AliveColumn(_network_agent.is_alive), _network_agent.availability_zone, _network_agent.binary, format_columns.DictColumn(_network_agent.configuration), _network_agent.created_at, _network_agent.description, _network_agent.ha_state, _network_agent.host, _network_agent.id, _network_agent.last_heartbeat_at, _network_agent.resources_synced, _network_agent.started_at, _network_agent.topic)

    def setUp(self):
        super(TestShowNetworkAgent, self).setUp()
        self.network_client.get_agent = mock.Mock(return_value=self._network_agent)
        self.cmd = network_agent.ShowNetworkAgent(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._network_agent.id]
        verifylist = [('network_agent', self._network_agent.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.get_agent.assert_called_once_with(self._network_agent.id)
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(len(list(self.data)), len(list(data)))
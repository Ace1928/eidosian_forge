from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNetworkAgent(TestNetworkAgent):
    network_agents = network_fakes.create_network_agents(count=2)

    def setUp(self):
        super(TestDeleteNetworkAgent, self).setUp()
        self.network_client.delete_agent = mock.Mock(return_value=None)
        self.cmd = network_agent.DeleteNetworkAgent(self.app, self.namespace)

    def test_network_agent_delete(self):
        arglist = [self.network_agents[0].id]
        verifylist = [('network_agent', [self.network_agents[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_agent.assert_called_once_with(self.network_agents[0].id, ignore_missing=False)
        self.assertIsNone(result)

    def test_multi_network_agents_delete(self):
        arglist = []
        for n in self.network_agents:
            arglist.append(n.id)
        verifylist = [('network_agent', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for n in self.network_agents:
            calls.append(call(n.id, ignore_missing=False))
        self.network_client.delete_agent.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_network_agents_delete_with_exception(self):
        arglist = [self.network_agents[0].id, 'unexist_network_agent']
        verifylist = [('network_agent', [self.network_agents[0].id, 'unexist_network_agent'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        delete_mock_result = [True, exceptions.CommandError]
        self.network_client.delete_agent = mock.Mock(side_effect=delete_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 network agents failed to delete.', str(e))
        self.network_client.delete_agent.assert_any_call(self.network_agents[0].id, ignore_missing=False)
        self.network_client.delete_agent.assert_any_call('unexist_network_agent', ignore_missing=False)
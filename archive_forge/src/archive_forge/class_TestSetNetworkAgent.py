from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkAgent(TestNetworkAgent):
    _network_agent = network_fakes.create_one_network_agent()

    def setUp(self):
        super(TestSetNetworkAgent, self).setUp()
        self.network_client.update_agent = mock.Mock(return_value=None)
        self.network_client.get_agent = mock.Mock(return_value=self._network_agent)
        self.cmd = network_agent.SetNetworkAgent(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self._network_agent.id]
        verifylist = [('network_agent', self._network_agent.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_agent.assert_called_once_with(self._network_agent, **attrs)
        self.assertIsNone(result)

    def test_set_all(self):
        arglist = ['--description', 'new_description', '--enable', self._network_agent.id]
        verifylist = [('description', 'new_description'), ('enable', True), ('disable', False), ('network_agent', self._network_agent.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'description': 'new_description', 'admin_state_up': True, 'is_admin_state_up': True}
        self.network_client.update_agent.assert_called_once_with(self._network_agent, **attrs)
        self.assertIsNone(result)

    def test_set_with_disable(self):
        arglist = ['--disable', self._network_agent.id]
        verifylist = [('enable', False), ('disable', True), ('network_agent', self._network_agent.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': False, 'is_admin_state_up': False}
        self.network_client.update_agent.assert_called_once_with(self._network_agent, **attrs)
        self.assertIsNone(result)
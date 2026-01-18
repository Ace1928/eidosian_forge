from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network_agent
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_all(self):
    arglist = ['--description', 'new_description', '--enable', self._network_agent.id]
    verifylist = [('description', 'new_description'), ('enable', True), ('disable', False), ('network_agent', self._network_agent.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'description': 'new_description', 'admin_state_up': True, 'is_admin_state_up': True}
    self.network_client.update_agent.assert_called_once_with(self._network_agent, **attrs)
    self.assertIsNone(result)
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_set_task_state(self):
    arglist = [self.share_server.id, '--task-state', 'migration_in_progress']
    verifylist = [('share_server', self.share_server.id), ('task_state', 'migration_in_progress')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.servers_mock.reset_task_state.assert_called_with(self.share_server, parsed_args.task_state)
    self.assertIsNone(result)
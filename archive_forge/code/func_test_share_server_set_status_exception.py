from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_set_status_exception(self):
    arglist = [self.share_server.id, '--status', 'active']
    verifylist = [('share_server', self.share_server.id), ('status', 'active')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.servers_mock.reset_state.side_effect = Exception()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
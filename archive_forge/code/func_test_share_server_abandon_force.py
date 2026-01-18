from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_abandon_force(self):
    arglist = [self.share_server.id, '--force']
    verifylist = [('share_server', [self.share_server.id]), ('force', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.servers_mock.unmanage.assert_called_with(self.share_server, force=True)
    self.assertIsNone(result)
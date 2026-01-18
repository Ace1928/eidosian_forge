from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_abandon_multiple(self):
    share_servers = manila_fakes.FakeShareServer.create_share_servers(count=2)
    arglist = [share_servers[0].id, share_servers[1].id]
    verifylist = [('share_server', [share_servers[0].id, share_servers[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.servers_mock.unmanage.call_count, len(share_servers))
    self.assertIsNone(result)
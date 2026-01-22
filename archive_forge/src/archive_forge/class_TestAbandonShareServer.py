from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestAbandonShareServer(TestShareServer):

    def setUp(self):
        super(TestAbandonShareServer, self).setUp()
        self.share_server = manila_fakes.FakeShareServer.create_one_server()
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.AbandonShareServer(self.app, None)
        self.data = tuple(self.share_server._info.values())
        self.columns = tuple(self.share_server._info.keys())

    def test_share_server_abandon_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_server_abandon(self):
        arglist = [self.share_server.id]
        verifylist = [('share_server', [self.share_server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.unmanage.assert_called_with(self.share_server)
        self.assertIsNone(result)

    def test_share_server_abandon_multiple(self):
        share_servers = manila_fakes.FakeShareServer.create_share_servers(count=2)
        arglist = [share_servers[0].id, share_servers[1].id]
        verifylist = [('share_server', [share_servers[0].id, share_servers[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.servers_mock.unmanage.call_count, len(share_servers))
        self.assertIsNone(result)

    def test_share_server_abandon_force(self):
        arglist = [self.share_server.id, '--force']
        verifylist = [('share_server', [self.share_server.id]), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.unmanage.assert_called_with(self.share_server, force=True)
        self.assertIsNone(result)

    def test_share_server_abandon_force_exception(self):
        arglist = [self.share_server.id]
        verifylist = [('share_server', [self.share_server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.servers_mock.unmanage.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_server_abandon_wait(self):
        arglist = [self.share_server.id, '--wait']
        verifylist = [('share_server', [self.share_server.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.servers_mock.unmanage.assert_called_with(self.share_server)
            self.assertIsNone(result)

    def test_share_server_abandon_wait_error(self):
        arglist = [self.share_server.id, '--wait']
        verifylist = [('share_server', [self.share_server.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_migration_start_with_new_share_network(self):
    """Test share server migration with new_share_network"""
    arglist = ['1234', 'host@backend', '--preserve-snapshots', 'False', '--writable', 'False', '--nondisruptive', 'False', '--new-share-network', self.new_share_network.id]
    verifylist = [('share_server', '1234'), ('host', 'host@backend'), ('preserve_snapshots', 'False'), ('writable', 'False'), ('nondisruptive', 'False'), ('new_share_network', self.new_share_network.id)]
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.57')
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.share_server.migration_start.assert_called_with('host@backend', 'False', 'False', 'False', self.new_share_network.id)
    self.assertEqual(result, ({}, {}))
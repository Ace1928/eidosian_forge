from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_adopt_wait(self):
    arglist = ['somehost@backend', self.share_network['id'], 'share_server_identifier', '--share-network-subnet', self.share_network_subnet['id'], '--wait']
    verifylist = [('host', 'somehost@backend'), ('share_network', self.share_network['id']), ('identifier', 'share_server_identifier'), ('share_network_subnet', self.share_network_subnet['id']), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=True):
        self.cmd.take_action(parsed_args)
        self.servers_mock.manage.assert_called_with(host='somehost@backend', share_network_id=self.share_network['id'], identifier='share_server_identifier', driver_options={}, share_network_subnet_id=self.share_network_subnet['id'])
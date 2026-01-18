from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_network_update_exception(self):
    share_network_name = 'share-network-name-' + uuid.uuid4().hex
    arglist = [self.share_network.id, '--name', share_network_name]
    verifylist = [('share_network', self.share_network.id), ('name', share_network_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.share_networks_mock.update.side_effect = Exception()
    with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.share_networks_mock.update.assert_called_once_with(self.share_network, name=parsed_args.name)
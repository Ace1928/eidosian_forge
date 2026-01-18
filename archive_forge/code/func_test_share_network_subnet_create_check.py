from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data(True, False)
def test_share_network_subnet_create_check(self, restart_check):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.70')
    self.share_networks_mock.share_network_subnet_create_check = mock.Mock(return_value=(200, {'compatible': True}))
    arglist = [self.share_network.id, '--check-only']
    verifylist = [('share_network', self.share_network.id), ('check_only', True)]
    if restart_check:
        arglist.append('--restart-check')
        verifylist.append(('restart_check', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.share_networks_mock.share_network_subnet_create_check.assert_called_once_with(share_network_id=self.share_network.id, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, reset_operation=restart_check)